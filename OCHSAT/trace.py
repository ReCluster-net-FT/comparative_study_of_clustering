"""
trace.py
========

Stores every event that occurs during the simulation.

The trace is later exported as trace.json so that
ns-3 / NetAnim can replay the complete simulation.

Examples of events

• member_to_ch
• aggregation
• ch_to_bs
• node_dead
• cluster_head_rotation
• reclustering
• graphsage_prediction
• ga_started
• ga_finished
"""

import json


class TraceLogger:

    def __init__(self):

        # Every simulation event is stored here.
        self.events = []

    # -------------------------------------------------------
    # Generic Event Logger
    # -------------------------------------------------------

    def log(self, round_number, event_type, **kwargs):
        """
        Stores one simulation event.

        Parameters
        ----------
        round_number : int

        event_type : str

        kwargs :
            Additional information describing the event.
        """

        event = {

            "round": int(round_number),

            "type": event_type

        }

        # Add additional information
        event.update(kwargs)

        self.events.append(event)

    # -------------------------------------------------------
    # Simulation Start
    # -------------------------------------------------------

    def simulation_start(
        self,
        protocol,
        nodes,
        base_station
    ):
        """
        Record the beginning of the simulation.

        This event stores the overall simulation configuration
        so that ns-3 can recreate the same scenario.
        """

        self.log(

            0,

            "simulation_start",

            protocol=str(protocol),

            node_count=int(nodes),

            bs_x=float(base_station[0]),

            bs_y=float(base_station[1])

        )


    def cluster_created(
        self,
        round_number,
        cluster_id,
        members
    ):

        self.log(
            round_number,
            "cluster_created",
            cluster=int(cluster_id),
            members=[int(x) for x in members]
        )


    def cluster_head_selected(
        self,
        round_number,
        cluster_id,
        node_id
    ):

        self.log(
            round_number,
            "cluster_head_selected",
            cluster=int(cluster_id),
            node=int(node_id)
        )


    def simulation_end(
        self,
        round_number
    ):

        self.log(
            round_number,
            "simulation_end"
        )

    def node_position(
        self,
        node_id,
        x,
        y
    ):
        self.log(
            0,
            "node_position",
            node=int(node_id),
            x=float(x),
            y=float(y)
        )

    # -------------------------------------------------------
    # Packet transmission
    # -------------------------------------------------------

    def member_to_ch(
        self,
        round_number,
        sender,
        receiver
    ):

        self.log(

            round_number,

            "member_to_ch",

            sender=int(sender),

            receiver=int(receiver)

        )

    # -------------------------------------------------------
    # Data aggregation
    # -------------------------------------------------------

    def aggregation(
        self,
        round_number,
        cluster_head
    ):

        self.log(

            round_number,

            "aggregation",

            cluster_head=int(cluster_head)

        )

    # -------------------------------------------------------
    # CH -> Base Station
    # -------------------------------------------------------

    def ch_to_bs(
        self,
        round_number,
        cluster_head
    ):

        self.log(

            round_number,

            "ch_to_bs",

            cluster_head=int(cluster_head)

        )

    # -------------------------------------------------------
    # Node Death
    # -------------------------------------------------------

    def node_dead(
        self,
        round_number,
        node
    ):

        self.log(

            round_number,

            "node_dead",

            node=int(node)

        )

    # -------------------------------------------------------
    # Cluster Head Rotation
    # -------------------------------------------------------

    def rotate_ch(
        self,
        round_number,
        cluster,
        old_ch,
        new_ch
    ):

        self.log(

            round_number,

            "cluster_head_rotation",

            cluster=int(cluster),

            old_cluster_head=int(old_ch),

            new_cluster_head=int(new_ch)

        )

    # -------------------------------------------------------
    # Re-clustering
    # -------------------------------------------------------

    def recluster(
        self,
        round_number
    ):

        self.log(

            round_number,

            "reclustering"

        )

    # -------------------------------------------------------
    # GraphSAGE Prediction
    # -------------------------------------------------------

    def graphsage_prediction(
        self,
        round_number,
        node,
        prediction
    ):

        self.log(

            round_number,

            "graphsage_prediction",

            node=int(node),

            prediction=str(prediction)

        )

    # -------------------------------------------------------
    # GA Started
    # -------------------------------------------------------

    def ga_started(
        self,
        round_number
    ):

        self.log(

            round_number,

            "ga_started"

        )

    # -------------------------------------------------------
    # GA Finished
    # -------------------------------------------------------

    def ga_finished(
        self,
        round_number
    ):

        self.log(

            round_number,

            "ga_finished"

        )

    # -------------------------------------------------------
    # Export JSON
    # -------------------------------------------------------

    def save(
        self,
        filename="trace.json"
    ):
        """
        Export the complete trace.
        """

        with open(filename, "w") as f:

            json.dump(

                self.events,

                f,

                indent=4

            )

        print(f"\nTrace exported to {filename}")

    # -------------------------------------------------------
    # Clear Trace
    # -------------------------------------------------------

    def clear(self):

        self.events.clear()
