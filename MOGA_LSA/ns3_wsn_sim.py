"""ns-3 WSN simulation driven by a cluster-head roster (from wsn_sim's MOGA
or MOGA+GraphSAGE selector, plus LSA routing -- see wsn_sim/ns3_export.py).

Reads a roster CSV (node_id, x, y, z, initial_energy, is_cluster_head,
relay_to), adds one explicit sink/base-station node at --sink-x/--sink-y,
and simulates routing: members -> nearest cluster head; cluster heads ->
`relay_to` (another head) if set, else direct to the sink. Node energy
sources are seeded from the roster's initial_energy. Produces a NetAnim
XML trace.

Uses ns-3's cppyy-based Python bindings (`from ns import ns`), so no C++
compilation of the scenario itself is needed -- only ns-3 core needs to be
built with --enable-python-bindings.

Each CSV row gets a local index 0..n-1, which is what indexes into the
ns-3 NodeContainer/Ipv4InterfaceContainer -- the CSV's node_id is only a
label, and must never be used as a container index.
"""
import argparse
import csv
import math
import os
import sys

NS3_BINDINGS_DIR = "/home/siddhartha/ns-allinone-3.45/ns-3.45/build/bindings/python"
if NS3_BINDINGS_DIR not in sys.path:
    sys.path.insert(0, NS3_BINDINGS_DIR)

try:
    from ns import ns
except ModuleNotFoundError:
    raise SystemExit(
        "ns3 Python module not found; build ns-3 with --enable-python-bindings "
        f"and check NS3_BINDINGS_DIR={NS3_BINDINGS_DIR}"
    )

SINK_ENERGY_J = 1000.0  # base station is mains-powered, not battery-constrained


def load_roster(csv_path, sink_x, sink_y):
    roster = []
    with open(csv_path) as f:
        for idx, row in enumerate(csv.DictReader(f)):
            roster.append(dict(
                idx=idx, node_id=int(row["node_id"]),
                x=float(row["x"]), y=float(row["y"]), z=float(row["z"]),
                initial_energy=float(row["initial_energy"]),
                is_cluster_head=bool(int(row["is_cluster_head"])),
                relay_to=int(row["relay_to"]),
                is_sink=False,
            ))
    sink_idx = len(roster)
    roster.append(dict(idx=sink_idx, node_id=-1, x=sink_x, y=sink_y, z=0.0,
                        initial_energy=SINK_ENERGY_J, is_cluster_head=False, relay_to=-1, is_sink=True))
    return roster, sink_idx


def nearest_cluster_head(member, heads):
    def dist(a, b):
        return math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
    return min(heads, key=lambda h: dist(member, h))


def build_and_run(roster, sink_idx, anim_path, sim_time):
    n = len(roster)
    nodes = ns.NodeContainer()
    nodes.Create(n)

    alloc = ns.ListPositionAllocator()
    for r in roster:
        alloc.Add(ns.Vector(r["x"], r["y"], r["z"]))
    mobility = ns.MobilityHelper()
    mobility.SetPositionAllocator(alloc)
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(nodes)

    wifi = ns.WifiHelper()
    wifi.SetStandard(ns.WIFI_STANDARD_80211b)
    phy = ns.YansWifiPhyHelper()
    channel = ns.YansWifiChannelHelper.Default()
    phy.SetChannel(channel.Create())
    mac = ns.WifiMacHelper()
    mac.SetType("ns3::AdhocWifiMac")
    devices = wifi.Install(phy, mac, nodes)

    stack = ns.InternetStackHelper()
    stack.Install(nodes)
    address = ns.Ipv4AddressHelper()
    address.SetBase(ns.Ipv4Address("10.1.0.0"), ns.Ipv4Mask("255.255.0.0"))
    interfaces = address.Assign(devices)

    # per-node energy source seeded from the roster, radio draws from it
    energy_sources = []
    for r in roster:
        src_helper = ns.BasicEnergySourceHelper()
        src_helper.Set("BasicEnergySourceInitialEnergyJ", ns.DoubleValue(r["initial_energy"]))
        src = src_helper.Install(nodes.Get(r["idx"]))
        energy_sources.append(src)
        radio_helper = ns.WifiRadioEnergyModelHelper()
        radio_helper.Install(ns.NetDeviceContainer(devices.Get(r["idx"])), src)

    sink = roster[sink_idx]
    heads = [r for r in roster if r["is_cluster_head"]]
    idx_by_node_id = {r["node_id"]: r for r in roster}

    # every node that can receive traffic (sink + cluster heads) gets a server
    port_by_idx = {}
    next_port = 8000
    for r in [sink] + heads:
        port_by_idx[r["idx"]] = next_port
        server = ns.UdpEchoServerHelper(next_port)
        apps = server.Install(nodes.Get(r["idx"]))
        apps.Start(ns.Seconds(0.0))
        apps.Stop(ns.Seconds(sim_time))
        next_port += 1

    def add_client(src_r, dst_r, start_offset):
        # this run is only for NetAnim topology visualization (the round-based
        # simulate.run() engine handles actual lifetime/energy analysis), and
        # the dataset's real per-node energy budgets are tiny (~0.3-0.8J) --
        # keep traffic light so nodes don't hit ns-3's energy-depletion assert
        # mid-animation.
        client = ns.UdpEchoClientHelper(interfaces.GetAddress(dst_r["idx"]).ConvertTo(), port_by_idx[dst_r["idx"]])
        client.SetAttribute("MaxPackets", ns.UintegerValue(5))
        client.SetAttribute("Interval", ns.TimeValue(ns.Seconds(2.0)))
        client.SetAttribute("PacketSize", ns.UintegerValue(128 if src_r["is_cluster_head"] else 96))
        capps = client.Install(nodes.Get(src_r["idx"]))
        capps.Start(ns.Seconds(1.0 + start_offset))
        capps.Stop(ns.Seconds(sim_time))

    # cluster heads relay to another head (relay_to) or direct to the sink
    for head in heads:
        dst = idx_by_node_id[head["relay_to"]] if head["relay_to"] != -1 else sink
        add_client(head, dst, start_offset=0.0)

    # members send to their nearest cluster head
    members = [r for r in roster if not r["is_cluster_head"] and not r["is_sink"]]
    for mi, member in enumerate(members):
        dst = nearest_cluster_head(member, heads) if heads else sink
        add_client(member, dst, start_offset=1.0 + 0.05 * mi)

    anim = ns.AnimationInterface(anim_path)
    anim.SetMaxPktsPerTraceFile(500000)
    for r in roster:
        node = nodes.Get(r["idx"])
        if r["is_sink"]:
            anim.UpdateNodeDescription(node, "Sink")
            anim.UpdateNodeColor(node, 255, 0, 0)
        elif r["is_cluster_head"]:
            anim.UpdateNodeDescription(node, f"ClusterHead-{r['node_id']}")
            anim.UpdateNodeColor(node, 0, 255, 0)
        else:
            anim.UpdateNodeDescription(node, f"Node-{r['node_id']}")
            anim.UpdateNodeColor(node, 0, 0, 255)

    ns.Simulator.Stop(ns.Seconds(sim_time))
    ns.Simulator.Run()
    ns.Simulator.Destroy()

    # AnimationInterface only writes the closing </anim> tag in its C++
    # destructor (StopAnimation() is private, not callable from Python).
    # Trigger just that one destructor now, before the mass cleanup below
    # that we need to dodge -- see the os._exit comment.
    del anim

    print(f"{n} nodes ({len(heads)} cluster heads + 1 sink), NetAnim trace written to {anim_path}")

    # cppyy JIT-compiles ns-3 destructor wrappers lazily; with this many
    # distinct helper/container objects (energy sources, app containers,
    # AnimationInterface, ...), letting Python's normal refcounting destroy
    # them here (i.e. on function return) reliably corrupts cppyy/cling's
    # JIT state and crashes (known cppyy/cling limitation at this object
    # count, unrelated to the simulation logic). Everything we care about
    # is already flushed to disk above, so exit the whole process
    # immediately instead of returning and triggering that cleanup.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-csv", default="ns3_nodes.csv")
    ap.add_argument("--anim-out", default="WSN_dataset_netanim.xml")
    ap.add_argument("--sim-time", type=float, default=30.0)
    ap.add_argument("--sink-x", type=float, default=50.0)
    ap.add_argument("--sink-y", type=float, default=50.0)
    args = ap.parse_args()

    roster, sink_idx = load_roster(args.nodes_csv, args.sink_x, args.sink_y)
    build_and_run(roster, sink_idx, args.anim_out, args.sim_time)
