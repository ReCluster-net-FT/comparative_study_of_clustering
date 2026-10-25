# ============================================================
# ga_router.py
# Centralized Deterministic CH-Only Multi-Hop Routing for WSN
#
# Communication model:
#
#   Ordinary Node -> Assigned CH only
#   Ordinary Node -> BS              NOT ALLOWED
#   Ordinary Node -> Ordinary Node   NOT ALLOWED
#   CH -> CH                         ALLOWED
#   CH -> BS                         ALLOWED
#
# Routing rule:
#
#   A CH can communicate with another CH if the distance is
#   <= max_tx_range.
#
#   A CH can communicate directly with BS if the distance is
#   <= max_tx_range.
#
#   Multi-hop routing uses BFS.
#
# IMPORTANT:
#
#   There is NO requirement that every next CH must be closer
#   to the BS.
#
# Runtime routing does NOT terminate the entire simulation if
# some surviving CHs become disconnected after node death.
#
# Disconnected CHs simply receive no route for that round.
#
# GA feasibility should be checked separately.
# ============================================================

import math
from collections import deque


class WSN_GA_Router:

    def __init__(
        self,
        df,
        cluster_heads,
        base_station,
        max_tx_range,
        packet_size,
        E_elec=None,
        E_fs=None,
        E_mp=None,
        E_DA=None,
        dist_matrix=None
    ):

        # ========================================================
        # Store inputs
        # ========================================================

        self.df = df.copy()

        self.cluster_heads = sorted(
            [int(ch) for ch in cluster_heads]
        )

        self.base_station = (
            float(base_station[0]),
            float(base_station[1])
        )

        self.max_tx_range = float(max_tx_range)
        self.packet_size = int(packet_size)

        # Energy parameters are accepted for compatibility.
        # Energy accounting is performed by simulation1.py.
        self.E_elec = E_elec
        self.E_fs = E_fs
        self.E_mp = E_mp
        self.E_DA = E_DA

        # Retained for compatibility.
        self.dist_matrix = dist_matrix

        # ========================================================
        # Coordinate lookup
        # ========================================================

        self.coordinates = {}

        self._build_coordinate_lookup()

        # ========================================================
        # Routing structures
        # ========================================================

        self.ch_graph = {}

        self.bs_reachable_chs = []

        self.next_hop = {}

        self.routes = {}

    # ============================================================
    # BUILD COORDINATE LOOKUP
    # ============================================================

    def _build_coordinate_lookup(self):

        columns = list(self.df.columns)

        node_id_col = None

        for col in [
            "Node_ID",
            "Node",
            "node",
            "NodeID",
            "node_id",
            "NODE_ID",
            "ID",
            "id"
        ]:
            if col in columns:
                node_id_col = col
                break

        x_col = None

        for col in [
            "X_Coordinate",
            "X",
            "x",
            "PosX",
            "pos_x",
            "position_x"
        ]:
            if col in columns:
                x_col = col
                break

        y_col = None

        for col in [
            "Y_Coordinate",
            "Y",
            "y",
            "PosY",
            "pos_y",
            "position_y"
        ]:
            if col in columns:
                y_col = col
                break

        if node_id_col is None:

            raise ValueError(
                "\nWSN_GA_Router: Could not find node ID column.\n"
                f"Available columns: {columns}\n"
                "Expected Node_ID."
            )

        if x_col is None or y_col is None:

            raise ValueError(
                "\nWSN_GA_Router: Could not find coordinate columns.\n"
                f"Available columns: {columns}\n"
                "Expected X_Coordinate and Y_Coordinate."
            )

        for _, row in self.df.iterrows():

            try:

                node_id = int(row[node_id_col])

                x = float(row[x_col])

                y = float(row[y_col])

            except (ValueError, TypeError) as exc:

                raise ValueError(
                    "\nInvalid node coordinate data.\n"
                    f"Node ID column : {node_id_col}\n"
                    f"X column       : {x_col}\n"
                    f"Y column       : {y_col}"
                ) from exc

            self.coordinates[node_id] = (x, y)

        missing_chs = [
            ch
            for ch in self.cluster_heads
            if ch not in self.coordinates
        ]

        if missing_chs:

            raise ValueError(
                "\nWSN_GA_Router: Coordinates not found for "
                f"selected CHs: {missing_chs}\n"
                f"Available node IDs: "
                f"{sorted(self.coordinates.keys())[:20]}"
            )

    # ============================================================
    # DISTANCE BETWEEN TWO SENSOR NODES
    # ============================================================

    def node_distance(self, node_a, node_b):

        node_a = int(node_a)
        node_b = int(node_b)

        if node_a not in self.coordinates:

            raise ValueError(
                f"Node {node_a} has no coordinates."
            )

        if node_b not in self.coordinates:

            raise ValueError(
                f"Node {node_b} has no coordinates."
            )

        xa, ya = self.coordinates[node_a]

        xb, yb = self.coordinates[node_b]

        return math.sqrt(
            (xa - xb) ** 2 +
            (ya - yb) ** 2
        )

    # ============================================================
    # COMPATIBILITY METHOD
    # ============================================================

    def distance(self, node_a, node_b):

        return self.node_distance(
            node_a,
            node_b
        )

    # ============================================================
    # DISTANCE FROM CH TO BASE STATION
    # ============================================================

    def bs_distance(self, ch):

        ch = int(ch)

        if ch not in self.coordinates:

            raise ValueError(
                f"CH {ch} has no coordinates."
            )

        x, y = self.coordinates[ch]

        bx, by = self.base_station

        return math.sqrt(
            (x - bx) ** 2 +
            (y - by) ** 2
        )

    # ============================================================
    # COMPATIBILITY METHOD
    # ============================================================

    def distance_to_bs(self, ch):

        return self.bs_distance(ch)

    # ============================================================
    # CHECK CH -> BS
    # ============================================================

    def can_reach_bs(self, ch):

        return (
            self.bs_distance(ch)
            <= self.max_tx_range
        )

    # ============================================================
    # CHECK CH -> CH
    # ============================================================

    def can_reach_ch(self, source, destination):

        source = int(source)
        destination = int(destination)

        if source == destination:
            return False

        if source not in self.cluster_heads:
            return False

        if destination not in self.cluster_heads:
            return False

        distance = self.node_distance(
            source,
            destination
        )

        return distance <= self.max_tx_range

    # ============================================================
    # BUILD CH-TO-CH GRAPH
    # ============================================================

    def build_ch_graph(self):

        graph = {
            ch: []
            for ch in self.cluster_heads
        }

        for i in range(
            len(self.cluster_heads)
        ):

            ch_a = self.cluster_heads[i]

            for j in range(
                i + 1,
                len(self.cluster_heads)
            ):

                ch_b = self.cluster_heads[j]

                if self.can_reach_ch(
                    ch_a,
                    ch_b
                ):

                    graph[ch_a].append(ch_b)

                    graph[ch_b].append(ch_a)

        # Deterministic ordering.
        for ch in graph:

            graph[ch].sort()

        self.ch_graph = graph

        return graph

    # ============================================================
    # PRINT CH CONNECTIVITY
    # ============================================================

    def print_ch_connectivity(self):

        if not self.ch_graph:

            self.build_ch_graph()

        print("\nCH-to-CH connectivity:")

        for ch in self.cluster_heads:

            print(
                f"  CH {ch:3d} -> "
                f"{self.ch_graph.get(ch, [])}"
            )

    # ============================================================
    # FIND BS-REACHABLE CHs
    # ============================================================

    def find_bs_reachable_chs(self):

        self.bs_reachable_chs = [
            ch
            for ch in self.cluster_heads
            if self.can_reach_bs(ch)
        ]

        print(
            "\nCHs directly reachable from BS:"
        )

        if not self.bs_reachable_chs:

            print("  NONE")

        else:

            for ch in self.bs_reachable_chs:

                print(
                    f"  CH {ch:3d} -> BS "
                    f"(distance = "
                    f"{self.bs_distance(ch):.4f} m)"
                )

        return self.bs_reachable_chs

    # ============================================================
    # BUILD NEXT-HOP TABLE
    #
    # Multi-source BFS starting from every CH that can directly
    # reach the BS.
    #
    # IMPORTANT:
    #
    # This function DOES NOT raise an exception if some CHs
    # are disconnected.
    #
    # It simply leaves those CHs without a next hop.
    # ============================================================

    def build_next_hop_table(self):

        if not self.ch_graph:

            self.build_ch_graph()

        bs_reachable_chs = \
            self.find_bs_reachable_chs()

        self.next_hop = {}

        # --------------------------------------------------------
        # No CH can directly reach BS.
        #
        # This is a runtime routing condition, not a Python
        # exception.
        # --------------------------------------------------------

        if not bs_reachable_chs:

            return self.next_hop

        queue = deque()

        visited = set()

        # --------------------------------------------------------
        # Multi-source BFS roots
        # --------------------------------------------------------

        for root in sorted(
            bs_reachable_chs
        ):

            self.next_hop[root] = None

            visited.add(root)

            queue.append(root)

        # --------------------------------------------------------
        # BFS
        # --------------------------------------------------------

        while queue:

            current = queue.popleft()

            for neighbor in self.ch_graph[current]:

                if neighbor in visited:
                    continue

                visited.add(neighbor)

                # Neighbor sends to current CH.
                self.next_hop[neighbor] = current

                queue.append(neighbor)

        return self.next_hop

    # ============================================================
    # BUILD ROUTE FOR ONE CH
    #
    # Example:
    #
    #     [81, 27, "BS"]
    #
    # or:
    #
    #     [18, 103, "BS"]
    #
    # If the CH cannot reach BS:
    #
    #     return None
    #
    # ============================================================

    def build_route(self, source_ch):

        source_ch = int(source_ch)

        if source_ch not in self.cluster_heads:

            raise ValueError(
                f"Node {source_ch} is not a selected CH."
            )

        if not self.next_hop:

            self.build_next_hop_table()

        # --------------------------------------------------------
        # No route exists.
        # --------------------------------------------------------

        if source_ch not in self.next_hop:

            return None

        route = [source_ch]

        current = source_ch

        visited = {source_ch}

        while True:

            next_ch = self.next_hop.get(
                current
            )

            # ----------------------------------------------------
            # Current CH directly reaches BS.
            # ----------------------------------------------------

            if next_ch is None:

                if not self.can_reach_bs(
                    current
                ):

                    return None

                route.append("BS")

                break

            # ----------------------------------------------------
            # Prevent loops.
            # ----------------------------------------------------

            if next_ch in visited:

                return None

            # ----------------------------------------------------
            # Destination must be selected CH.
            # ----------------------------------------------------

            if next_ch not in self.cluster_heads:

                return None

            # ----------------------------------------------------
            # Verify physical CH-to-CH distance.
            # ----------------------------------------------------

            distance = self.node_distance(
                current,
                next_ch
            )

            if distance > self.max_tx_range:

                return None

            route.append(next_ch)

            visited.add(next_ch)

            current = next_ch

        self.routes[source_ch] = route

        return route

    # ============================================================
    # BUILD ALL ROUTES
    #
    # IMPORTANT:
    #
    # This returns routes for every CH that can reach BS.
    #
    # A disconnected CH is stored as:
    #
    #     routes[ch] = None
    #
    # The simulation can therefore continue.
    # ============================================================

    def build_all_routes(self):

        self.build_ch_graph()

        self.print_ch_connectivity()

        self.build_next_hop_table()

        self.routes = {}

        # --------------------------------------------------------
        # Build routes independently.
        # --------------------------------------------------------

        for ch in self.cluster_heads:

            route = self.build_route(ch)

            self.routes[ch] = route

        # --------------------------------------------------------
        # Print runtime routing status.
        # --------------------------------------------------------

        unreachable = [
            ch
            for ch in self.cluster_heads
            if self.routes.get(ch) is None
        ]

        if unreachable:

            print(
                "\nWARNING: The following CHs have no "
                "BS route in this round:"
            )

            print(
                f"  {unreachable}"
            )

        else:

            print(
                "\nAll selected CHs have valid BS routes."
            )

        # IMPORTANT:
        # simulation1.py expects exactly two values.
        return (
            self.next_hop,
            self.routes
        )

    # ============================================================
    # VALIDATE ONE ROUTE
    # ============================================================

    def validate_route(self, route):

        if route is None:
            return False

        if len(route) < 2:
            return False

        # --------------------------------------------------------
        # Final element must be BS.
        # --------------------------------------------------------

        if route[-1] != "BS":
            return False

        # --------------------------------------------------------
        # Every element before BS must be a selected CH.
        # --------------------------------------------------------

        for ch in route[:-1]:

            if ch not in self.cluster_heads:

                return False

        # --------------------------------------------------------
        # Validate CH -> CH hops.
        # --------------------------------------------------------

        for i in range(
            len(route) - 2
        ):

            current = route[i]

            next_ch = route[i + 1]

            if not self.can_reach_ch(
                current,
                next_ch
            ):

                return False

        # --------------------------------------------------------
        # Validate final CH -> BS.
        # --------------------------------------------------------

        final_ch = route[-2]

        if not self.can_reach_bs(
            final_ch
        ):

            return False

        return True

    # ============================================================
    # VALIDATE ALL ROUTES
    #
    # IMPORTANT:
    #
    # Returns False if ANY CH has no route.
    #
    # This is useful for GA validation/debugging.
    #
    # It does NOT terminate the simulation by itself.
    # ============================================================

    def validate_all_routes(self):

        if not self.routes:

            self.build_all_routes()

        invalid_routes = []

        for ch in self.cluster_heads:

            route = self.routes.get(ch)

            if not self.validate_route(
                route
            ):

                invalid_routes.append(
                    (ch, route)
                )

        if invalid_routes:

            print(
                "\nINVALID / UNREACHABLE ROUTES:"
            )

            for ch, route in invalid_routes:

                print(
                    f"  CH {ch}: {route}"
                )

            return False

        return True

    # ============================================================
    # CHECK WHETHER ALL CHs ARE BS-CONNECTED
    #
    # This is the strict feasibility test that should be used
    # by GA / CH selection.
    # ============================================================

    def is_fully_bs_connected(self):

        if not self.ch_graph:

            self.build_ch_graph()

        if not self.next_hop:

            self.build_next_hop_table()

        for ch in self.cluster_heads:

            if ch not in self.next_hop:

                return False

        return True

    # ============================================================
    # PRINT ROUTING TABLE
    # ============================================================

    def print_routes(self):

        print(
            "\n============================================================"
        )

        print(
            "                 CH ROUTING TABLE"
        )

        print(
            "============================================================"
        )

        for ch in self.cluster_heads:

            route = self.routes.get(ch)

            if route is None:

                print(
                    f"  CH {ch:3d}: NO ROUTE"
                )

                continue

            route_string = " -> ".join(
                str(node)
                for node in route
            )

            print(
                f"  CH {ch:3d}: "
                f"{route_string}"
            )

        print(
            "============================================================"
        )

    # ============================================================
    # RUN COMPLETE ROUTING PROCESS
    #
    # NOTE:
    #
    # This method is retained for compatibility.
    #
    # It does NOT raise simply because some CH has no route.
    # ============================================================

    def run(self):

        print(
            "\n============================================================"
        )

        print(
            "             GA CH-ONLY ROUTING"
        )

        print(
            "============================================================"
        )

        print(
            f"Selected CHs       : "
            f"{self.cluster_heads}"
        )

        print(
            f"Base station       : "
            f"{self.base_station}"
        )

        print(
            f"Communication range: "
            f"{self.max_tx_range:.4f} m"
        )

        (
            next_hop_table,
            routes
        ) = self.build_all_routes()

        self.next_hop = next_hop_table
        self.routes = routes

        self.print_routes()

        if self.is_fully_bs_connected():

            print(
                "\nRouting feasibility : VALID"
            )

        else:

            print(
                "\nRouting feasibility : "
                "PARTIALLY CONNECTED"
            )

            print(
                "Disconnected CH aggregate packets "
                "will be lost for this round."
            )

        return (
            self.next_hop,
            self.routes
        )


# ============================================================
# END OF ga_router.py
# ============================================================
