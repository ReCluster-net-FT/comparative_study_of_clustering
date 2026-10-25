"""Build the plain-language MOGA+LSA pipeline explainer PDF for the team."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

INK = colors.HexColor("#12181F")
TEAL = colors.HexColor("#1B6E77")
PURPLE = colors.HexColor("#7A4FB5")
GREY = colors.HexColor("#57626C")
RULE = colors.HexColor("#DDE2E7")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold",
                    fontSize=21, textColor=INK, spaceAfter=4, alignment=0)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontName="Helvetica",
                     fontSize=11, textColor=GREY, spaceAfter=16)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=13.5, textColor=TEAL, spaceBefore=15, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                    fontSize=11, textColor=INK, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontName="Helvetica",
                      fontSize=10, leading=15.5, textColor=INK,
                      alignment=TA_JUSTIFY, spaceAfter=7)
CAP = ParagraphStyle("CAP", parent=ss["Normal"], fontName="Helvetica-Oblique",
                     fontSize=8.5, textColor=GREY, spaceBefore=3, spaceAfter=10)
NOTE = ParagraphStyle("NOTE", parent=BODY, fontSize=9.5, leading=14.5,
                      textColor=GREY, leftIndent=10, borderPadding=0)

W = 16.8 * cm


def h2(t): return Paragraph(t, H2)
def h3(t): return Paragraph(t, H3)
def p(t): return Paragraph(t, BODY)
def cap(t): return Paragraph(t, CAP)


def fig(path, width=W):
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    return Image(path, width=width, height=width * ih / iw)


def bullets(items, style=BODY):
    return ListFlowable([ListItem(Paragraph(i, style), leftIndent=12)
                         for i in items],
                        bulletType="bullet", start="circle",
                        leftIndent=14, bulletFontSize=7, spaceAfter=8)


story = []

story += [
    Paragraph("MOGA+LSA: How the Pipeline Works", H1),
    Paragraph("A plain-language walkthrough of the cluster-head selection and "
              "routing pipeline, and the GraphSAGE variant we compared it against.", SUB),
]

# ---------------------------------------------------------------- 1
story += [
    h2("1. The problem we are solving"),
    p("A wireless sensor network is a field of battery-powered sensor nodes that "
      "need to get their readings back to a base station (also called the "
      "<b>sink</b>). Radio transmission is what drains the batteries, and the "
      "cost grows very fast with distance, so having all 100 nodes shout "
      "directly at the base station would kill the network quickly."),
    p("The fix is <b>clustering</b>. A small number of nodes are promoted to "
      "<b>cluster heads</b> (CHs) each round. Ordinary nodes send their reading "
      "a short distance to their nearest cluster head, the cluster head merges "
      "everything it received into one message, and only the cluster heads pay "
      "the expensive long-distance transmission to the sink."),
    p("That leaves two questions to answer every round, and the two halves of "
      "this pipeline answer one each:"),
    bullets([
        "<b>Which nodes should be cluster heads?</b> &mdash; answered by MOGA, "
        "a genetic algorithm.",
        "<b>How should those cluster heads get their data to the sink?</b> "
        "&mdash; answered by LSA, a separate search algorithm.",
    ]),
]

# ---------------------------------------------------------------- 2
story += [
    h2("2. The pipeline at a glance"),
    fig("fig_pipeline.png"),
    cap("Figure 1. One simulation round. Everything inside repeats thousands of "
        "times until the nodes run out of energy."),
    p("The important thing to notice is that this whole sequence runs "
      "<b>every single round</b>. Cluster heads are not chosen once at the "
      "start and kept forever &mdash; they are re-chosen from scratch each "
      "round, because batteries drain and a node that was a great cluster head "
      "earlier may be nearly flat now."),
]

story.append(PageBreak())

# ---------------------------------------------------------------- 3
story += [
    h2("3. Step one: load the network"),
    p("We take 100 real sensor records from the Kaggle WSN dataset. Each node "
      "keeps its real X/Y position on a 100 m &times; 100 m field, its real "
      "starting battery level (rescaled so the average node starts at 0.5 "
      "joules), and eleven real sensor attributes such as signal strength, "
      "noise level, temperature and packet loss rate."),
    p("The sink is placed at the centre of the sampled nodes. Each random seed "
      "draws a different 100-node sample from the dataset, which is how we get "
      "variation between runs."),
]

# ---------------------------------------------------------------- 4
story += [
    h2("4. Step two: MOGA picks the cluster heads"),
    p("MOGA stands for <b>Multi-Objective Genetic Algorithm</b>. A genetic "
      "algorithm is a trial-and-error search loosely modelled on evolution: "
      "try a batch of candidate answers, keep the good ones, combine them to "
      "make new ones, add a little randomness, and repeat until the batch has "
      "improved."),

    h3("The vocabulary"),
    bullets([
        "<b>Chromosome</b> &mdash; one candidate answer. Here, one chromosome "
        "is one proposed set of about 10 cluster heads.",
        "<b>Population</b> &mdash; the batch of chromosomes being tried at "
        "once. We use 20.",
        "<b>Generation</b> &mdash; one pass of score / keep the best / breed / "
        "mutate. We run 10 generations per round.",
        "<b>Fitness</b> &mdash; a single number scoring how good a chromosome "
        "is. In our setup, <b>lower is better</b>.",
    ]),
    p("A common point of confusion: the 20 and the 10 are different counts "
      "stacked on top of each other. There are <b>20 competing proposals</b>, "
      "and <b>each one of those proposals contains its own 10 cluster "
      "heads</b>. The very first 20 proposals are built by picking 10 node IDs "
      "completely at random (no duplicates within one proposal). There is no "
      "cleverness at that starting point &mdash; the intelligence only comes "
      "from what happens next."),
]

story += [
    fig("fig_moga_loop.png"),
    cap("Figure 2. What MOGA does inside a single round."),
]

story.append(PageBreak())

story += [
    h3("The fitness formula, in words"),
    p("Each proposal is scored on three things added together:"),
    bullets([
        "<b>Member-to-CH distance</b> &mdash; how far each ordinary node is "
        "from its nearest cluster head, summed up. Smaller is better.",
        "<b>CH-to-sink distance</b> &mdash; how far the cluster heads are from "
        "the base station. Smaller is better.",
        "<b>Residual energy of the cluster heads</b> &mdash; how much battery "
        "the chosen cluster heads still have. This one is <b>subtracted</b>, "
        "because more battery is better and we are minimising the total.",
    ]),
    p("All three are normalised onto a 0&ndash;1 scale before being added, so "
      "that no single term dominates the score just because it happens to be "
      "measured in bigger units. In plain terms the formula is asking: "
      "<i>pick cluster heads that are close to their members, close to the "
      "base station, and still have plenty of battery.</i>"),

    h3("How breeding (crossover) works"),
    p("Take two surviving proposals and cut both at the same random point, "
      "then glue the front of one to the back of the other. With shortened "
      "6-head lists for readability:"),
]

cross = Table([
    ["Parent A", "12, 45, 7", "|", "88, 3, 61"],
    ["Parent B", "9, 22, 50", "|", "4, 77, 15"],
    ["Child", "12, 45, 7", "+", "4, 77, 15"],
], colWidths=[2.4 * cm, 4.2 * cm, 0.8 * cm, 4.2 * cm])
cross.setStyle(TableStyle([
    ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
    ("FONT", (1, 0), (-1, -1), "Courier", 9.5),
    ("TEXTCOLOR", (0, 2), (-1, 2), TEAL),
    ("FONT", (0, 2), (0, 2), "Helvetica-Bold", 9),
    ("LINEABOVE", (0, 2), (-1, 2), 0.7, RULE),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("ALIGN", (2, 0), (2, -1), "CENTER"),
]))
story += [cross, Spacer(1, 10)]

story += [
    p("The idea is that two proposals which both scored reasonably well "
      "probably each contain some genuinely good cluster-head picks mixed with "
      "some mediocre ones. Splicing them gives the child a chance of "
      "inheriting the good parts of both. Two housekeeping rules apply: if the "
      "splice produces a duplicate node, the duplicate is dropped, and if that "
      "leaves fewer than 10 cluster heads, random nodes are added to pad it "
      "back up."),
    p("<b>Mutation</b> then gives each child a 20% chance of having one of its "
      "cluster heads swapped for a different random node. This keeps a bit of "
      "fresh randomness in the population so the search does not get stuck "
      "refining the same narrow set of ideas."),
    p("After 10 generations of this, the single best-scoring proposal wins and "
      "its 10 nodes become the actual cluster heads for the round. The other "
      "19 are discarded &mdash; they existed only to help the search."),
]

story.append(PageBreak())

# ---------------------------------------------------------------- 5
story += [
    h2("5. Step three: LSA picks the routes"),
    p("LSA stands for <b>Lightning Search Algorithm</b>. It belongs to the "
      "same family as the genetic algorithm &mdash; start with random "
      "candidate answers, improve them over repeated iterations &mdash; but it "
      "is modelled on how a lightning strike probes for the easiest path to "
      "the ground rather than on evolution."),
    p("Its job here is different from MOGA's. MOGA has already decided "
      "<i>who</i> the cluster heads are. LSA decides <i>how their data gets "
      "home</i>. A cluster head does not have to transmit straight to the "
      "sink; it may be cheaper to hand the data to another cluster head that "
      "is closer to the sink and let that one forward it. This is the same "
      "logic as passing a message to a friend nearer the destination instead "
      "of shouting all the way there yourself, because shouting a long "
      "distance costs disproportionately more battery."),
    p("LSA searches over 30 iterations with 15 candidate routing plans, "
      "balancing three things at once: how much data actually arrives, how "
      "much battery the plan costs, and how long it takes."),
]

# ---------------------------------------------------------------- 6
story += [
    h2("6. Step four: energy is deducted"),
    p("Now the round actually happens and everyone pays for it. The cost is "
      "not a flat fee &mdash; it depends on how many bits were sent and how "
      "far they travelled:"),
    bullets([
        "<b>Sending</b> costs a fixed electronics charge per bit "
        "(50 nJ/bit), plus an amplifier charge that grows with "
        "distance<super>2</super> for short hops and distance<super>4</super> "
        "for long hops (the cutoff is about 87.7 m).",
        "<b>Receiving</b> costs only the electronics charge (50 nJ/bit). "
        "Distance does not matter when receiving.",
        "<b>Aggregating</b> costs a cluster head an extra 5 nJ/bit to merge "
        "everything its members sent into one outgoing message.",
    ]),
    p("So in one round an ordinary member pays once to transmit. A cluster "
      "head pays to receive every member's message, pays to aggregate them, "
      "then pays to transmit the result onward. A relaying cluster head pays "
      "to receive and then to transmit again. Any node whose battery reaches "
      "zero is marked dead and takes no further part in the simulation."),
    p("This distance-squared and distance-to-the-fourth behaviour is exactly "
      "why cluster-head placement matters so much, and it is worth keeping in "
      "mind for section 8."),
]

# ---------------------------------------------------------------- 7
story += [
    h2("7. Step five: repeat and record"),
    p("Steps two to four repeat every round, up to 13,000 rounds in our "
      "longest run. After each round we record how many nodes are still "
      "alive, the average battery remaining, the delivery rate and the "
      "average delay. Plotting the alive-node count over time is what gives us "
      "the three standard lifetime milestones:"),
    bullets([
        "<b>FND</b> (First Node Death) &mdash; the round the first node dies.",
        "<b>HND</b> (Half Node Death) &mdash; the round half the network is dead.",
        "<b>LND</b> (Last Node Death) &mdash; the round the final node dies.",
    ]),
]

story.append(PageBreak())

# ---------------------------------------------------------------- 8
story += [
    h2("8. The GraphSAGE variant"),
    p("Everything above describes the stock protocol. The variant we compared "
      "it against changes exactly one thing."),
    p("<b>GraphSAGE</b> is a neural network designed to work on data that is "
      "connected in a network rather than laid out in a flat table. It "
      "produces an <b>embedding</b> for each node: a list of 16 numbers that "
      "acts like a fingerprint, summarising not just the node itself but also "
      "what its neighbours look like."),

    h3("How it is trained"),
    bullets([
        "First we build a graph by connecting each node to its 8 physically "
        "nearest neighbours on the map. This graph only decides who counts as "
        "whose neighbour; separately, each node carries 14 numbers (position, "
        "battery and the eleven sensor readings) that the network reads.",
        "Then we train the network on a puzzle: <i>given two nodes' "
        "fingerprints, can you tell whether they are actually neighbours?</i> "
        "Real neighbours are the positive examples; random pairs are the "
        "negative examples.",
        "Nobody ever tells the network what a &ldquo;correct&rdquo; "
        "fingerprint looks like, so this is called <b>unsupervised</b> "
        "training. Getting good at the puzzle forces similar nodes to end up "
        "with similar fingerprints, which is the useful part.",
        "Training runs for 100 passes over the data, once at the very start of "
        "the simulation. The model is then frozen and the same fingerprints "
        "are reused for all 13,000 rounds.",
    ]),
    fig("fig_graphsage.png"),
    cap("Figure 3. The single point at which GraphSAGE enters the pipeline."),
    p("The only change in the pipeline is in the first term of the fitness "
      "formula. Instead of measuring member-to-cluster-head distance on the "
      "map, we measure it between the 16-number fingerprints. The genetic "
      "algorithm, the LSA routing, the other two fitness terms and the energy "
      "model are all completely unchanged, so any difference in the results "
      "comes from this substitution alone."),
]

# ---------------------------------------------------------------- 9
story += [
    h2("9. What we found"),
    p("The two methods performed essentially the same. Over 3,000 rounds no "
      "node died in either version, and the difference in remaining battery "
      "was 0.077% &mdash; smaller than the variation between random seeds, and "
      "not statistically significant (p = 0.123)."),
    p("Over the extended 13,000-round run the two lifetime milestones moved in "
      "opposite directions: FND got 3.7% worse with GraphSAGE while HND got "
      "0.9% better. LND was never reached by either version."),
]

res = Table([
    ["Metric", "MOGA+LSA", "MOGA+LSA + GraphSAGE"],
    ["Residual energy at 3,000 rounds", "0.33127 J", "0.33102 J  (p = 0.123)"],
    ["FND", "6,700 rounds", "6,450 rounds  (-3.7%)"],
    ["HND", "8,800 rounds", "8,875 rounds  (+0.9%)"],
    ["LND", "not reached", "not reached"],
], colWidths=[6.4 * cm, 4.6 * cm, 5.8 * cm])
res.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
    ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("BACKGROUND", (0, 0), (-1, 0), TEAL),
    ("GRID", (0, 0), (-1, -1), 0.5, RULE),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 7),
]))
story += [res, Spacer(1, 12)]

story += [
    p("<b>Why the null result is interesting rather than disappointing.</b> "
      "Recall from section 6 that transmission energy depends on "
      "<i>physical</i> distance. The member-to-CH distance term is therefore "
      "not an arbitrary measure of closeness &mdash; it is a direct stand-in "
      "for the energy the formula is trying to save. Replacing it with a "
      "learned fingerprint distance cuts the fitness formula loose from the "
      "energy model, and the extra structural information does not make up for "
      "what was lost."),
    p("That is a useful finding for the wider study: it suggests graph "
      "embeddings help when they are <b>added alongside</b> physically "
      "meaningful terms, rather than <b>substituted in place of</b> them. A "
      "second possible explanation is the training signal &mdash; our model "
      "learns which nodes look alike, but is never shown whether a given "
      "cluster-head choice actually shortened the network's life."),
]

# ---------------------------------------------------------------- 10
story += [h2("10. Where the code lives")]

code = Table([
    ["wsn_sim/wsn_dataset.py", "Loads the Kaggle dataset, samples 100 nodes"],
    ["wsn_sim/moga.py", "The genetic algorithm, fitness, crossover, mutation"],
    ["wsn_sim/lsa.py", "Lightning Search Algorithm routing"],
    ["wsn_sim/network.py", "Radio energy constants and network state"],
    ["wsn_sim/simulate.py", "The round loop and metric logging"],
    ["wsn_sim/graphsage_embed.py", "GraphSAGE model and its training"],
    ["wsn_sim/moga_embedding.py", "Wires the embeddings into MOGA"],
    ["run_compare.py", "3 seeds x 3,000 rounds comparison"],
    ["run_extended_lifetime.py", "1 seed x 13,000 rounds, for FND/HND/LND"],
    ["ns3_wsn_sim.py", "Exports the ns-3 / NetAnim packet-level traces"],
])
code.setStyle(TableStyle([
    ("FONT", (0, 0), (0, -1), "Courier-Bold", 9),
    ("FONT", (1, 0), (1, -1), "Helvetica", 9),
    ("TEXTCOLOR", (0, 0), (0, -1), PURPLE),
    ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]), )
story += [code, Spacer(1, 14)]

story += [
    p("<b>To reproduce the numbers in section 9:</b> run "
      "<font face='Courier'>python run_compare.py</font> for the 3-seed "
      "comparison and <font face='Courier'>python run_extended_lifetime.py</font> "
      "for the lifetime milestones. Both write their results to JSON files that "
      "the plotting scripts read."),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(2.4 * cm, 1.3 * cm, "MOGA+LSA pipeline walkthrough")
    canvas.drawRightString(A4[0] - 2.4 * cm, 1.3 * cm, f"page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(2.4 * cm, 1.75 * cm, A4[0] - 2.4 * cm, 1.75 * cm)
    canvas.restoreState()


doc = SimpleDocTemplate(
    "MOGA_LSA_pipeline_explained.pdf", pagesize=A4,
    leftMargin=2.4 * cm, rightMargin=2.4 * cm,
    topMargin=2.0 * cm, bottomMargin=2.2 * cm,
    title="MOGA+LSA: How the Pipeline Works",
    author="Team 26",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("saved MOGA_LSA_pipeline_explained.pdf")
