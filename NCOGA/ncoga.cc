
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>
#include <limits>
#include <cstdlib>
#include <string>
#include <thread>
#include <chrono>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace ns3;
#include <streambuf>

class TeeStreamBuf : public std::streambuf
{
public:
  TeeStreamBuf (std::streambuf *sb1, std::streambuf *sb2)
    : m_sb1 (sb1), m_sb2 (sb2) {}

protected:
  virtual int
  overflow (int c) override
  {
    if (c == EOF)
      {
        return !EOF;
      }
    int r1 = m_sb1->sputc (static_cast<char> (c));
    int r2 = m_sb2->sputc (static_cast<char> (c));
    return (r1 == EOF || r2 == EOF) ? EOF : c;
  }

  virtual int
  sync () override
  {
    int r1 = m_sb1->pubsync ();
    int r2 = m_sb2->pubsync ();
    return (r1 == 0 && r2 == 0) ? 0 : -1;
  }

private:
  std::streambuf *m_sb1;
  std::streambuf *m_sb2;
};

NS_LOG_COMPONENT_DEFINE ("NCOGA");

static std::ofstream g_resultsTxt;
static TeeStreamBuf *g_teeBuf = nullptr;
static std::streambuf *g_origClogBuf = nullptr;

static const double AREA_X = 100.0;
static const double AREA_Y = 100.0;
static const uint32_t NUM_NODES = 100;
static const double SINK_X = AREA_X / 2.0;
static const double SINK_Y = AREA_Y / 2.0;

static const double E0 = 0.5;
static const double E_SUP = 1.0;
static const double E_ADV = 2.0;
static const double P_NORMAL = 0.7;
static const double P_ADVANCED = 0.2;
static const double P_SUPER = 0.1;

static const double E_ELEC = 50e-9;
static const double E_FS = 10e-12;
static const double E_MP = 0.0013e-12;
static const double E_DA = 5e-9;
static const double PACKET_SIZE = 2000.0;
static const double D0 = std::sqrt (E_FS / E_MP);

static const double CHP_INIT = 0.05;
static const uint32_t POP_SIZE = 30;
static const uint32_t NUM_GENERATIONS = 30;
static const double PC_MIN = 0.5;
static const double PC_MAX = 1.0;
static const double PM = 0.031;

static const double W_FP1 = 1.0 / 6.0;
static const double W_FP2 = 1.0 / 6.0;
static const double W_FP3 = 1.0 / 6.0;
static const double W_FP4 = 1.0 / 6.0;
static const double W_FP5 = 1.0 / 6.0;
static const double W_FP6 = 1.0 / 6.0;

static const uint32_t MAX_ROUNDS = 30000;

static const double HARD_THRESHOLD = 0.5;
static const double SOFT_THRESHOLD = 0.01;
static const double SENSED_DRIFT_STD = 0.05;
static const double SENSED_INIT_MIN = 0.0;
static const double SENSED_INIT_MAX = 1.0;

static bool ENABLE_GRAPHSAGE = false;
static bool ENABLE_MLP = false;
static bool ENABLE_GRAPHSAGE_DATA = false;
static bool GRAPHSAGE_TRAINING_MODE = false;
static uint32_t SIMULATION_SEED = 1;
static uint32_t GRAPHSAGE_K = 250;
static double EXPLORATION_FRACTION = 0.0;
static bool GRAPHSAGE_DATA_MODE = false;
static std::string GRAPHSAGE_MODEL = "graphsage_model.pt";
static std::string MLP_MODEL = "mlp_model.pt";
static std::string GRAPHSAGE_INFERENCE_SCRIPT = "infer_graphsage.py";
static std::string GRAPHSAGE_PYTHON = "myEnv/bin/python3.12";
static std::string OUTPUT_PREFIX;
static double GRAPH_SAGE_WEIGHT = 0.0;
static double GRAPH_FITNESS_MEAN = 0.0;
static double GRAPH_FITNESS_STD = 1.0;
static double GRAPH_UTILITY_MEAN = 0.0;
static double GRAPH_UTILITY_STD = 1.0;
enum NodeType { NORMAL, ADVANCED, SUPER };

struct SensorNode
{
  uint32_t id;
  double x, y;
  double initialEnergy;
  double energy;
  bool alive;
  NodeType type;
  bool isCH;
  bool isRelayCH;
  int clusterId;
  uint32_t roundsSinceCH;
  double sensedValue;
  double lastSentValue;
  bool hasReportedOnce;
  uint32_t deathRound;
  bool everWasCH;
  bool everWasRelayCH;
  double relayEnergyThisRound;
};

static std::mt19937 g_rng (12345);

struct GraphSageSnapshot
{
  uint32_t seed;
  uint32_t round;
  uint32_t targetNode;
  double distanceToSink;
  double residualRatio;
  int normalType;
  int advancedType;
  int superType;
  double roundsSinceCHNormalized;
  uint32_t neighborCount;
  double localDensity;
  double energyAtSelection;
  std::vector<uint32_t> cluster;
  std::vector<double> clusterEnergyAtSelection;
  std::vector<double> featureNodes;
  std::vector<uint32_t> featureEdges;
  double sampleWeight;
  double utilityEnergy;
  double utilityCluster;
  double utilityClusterEnergy;
  double relayCost;
  double fitness;
};

static double
RandUniform ()
{
  static std::uniform_real_distribution<double> dist (0.0, 1.0);
  return dist (g_rng);
}

static std::vector<SensorNode> g_nodes;
static uint32_t g_round = 0;
static uint32_t g_firstNodeDeadRound = 0;
static uint32_t g_halfNodesDeadRound = 0;
static uint32_t g_lastNodeDeadRound = 0;
static bool g_fndRecorded = false;
static bool g_hndRecorded = false;
static bool g_lndRecorded = false;
static uint64_t g_totalPacketsToSink = 0;
static std::ofstream g_csv;
static std::ofstream g_aliveCsv;
static std::ofstream g_graphNodesCsv;
static std::ofstream g_graphStateCsv;
static std::ofstream g_graphEdgesCsv;
static std::ofstream g_summaryCsv;
static std::vector<GraphSageSnapshot> g_pendingGraphSnapshot;
static std::vector<GraphSageSnapshot> g_pendingSamples;
static double g_sumChCount = 0.0;
static uint32_t g_roundsExecuted = 0;

static std::string
OutputFile (const std::string &suffix)
{
  return OUTPUT_PREFIX.empty () ? suffix : OUTPUT_PREFIX + "_" + suffix;
}

static std::vector<double> g_distMatrix;
static std::vector<double> g_distToSinkCache;
static std::vector<double> g_graphSageScores;
static int g_graphSageSocket = -1;
static int g_graphSagePort = -1;

static double
Dist (double x1, double y1, double x2, double y2)
{
  return std::sqrt ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

static inline double
DistNodes (uint32_t idA, uint32_t idB)
{
  return g_distMatrix[static_cast<size_t> (idA) * NUM_NODES + idB];
}

static double
DistToSink (const SensorNode &n)
{
  return g_distToSinkCache[n.id];
}

static void
BuildDistanceCache ()
{
  g_distMatrix.assign (static_cast<size_t> (NUM_NODES) * NUM_NODES, 0.0);
  g_distToSinkCache.assign (NUM_NODES, 0.0);
  for (uint32_t i = 0; i < NUM_NODES; ++i)
    {
      g_distToSinkCache[i] = Dist (g_nodes[i].x, g_nodes[i].y, SINK_X, SINK_Y);
      for (uint32_t j = i; j < NUM_NODES; ++j)
        {
          double d = (i == j) ? 0.0
                               : Dist (g_nodes[i].x, g_nodes[i].y,
                                       g_nodes[j].x, g_nodes[j].y);
          g_distMatrix[static_cast<size_t> (i) * NUM_NODES + j] = d;
          g_distMatrix[static_cast<size_t> (j) * NUM_NODES + i] = d;
        }
    }
}

static double
Etx (double z, double d)
{
  if (d <= D0)
    {
      return z * E_ELEC + z * E_FS * d * d;
    }
  else
    {
      return z * E_ELEC + z * E_MP * d * d * d * d;
    }
}

static double
Erx (double z)
{
  return z * E_ELEC;
}

static double
Edx (double z, double numPackets)
{
  return numPackets * z * E_DA;
}

static void
DeployNodes ()
{
  g_nodes.clear ();
  g_nodes.resize (NUM_NODES);

  const uint32_t numNormal = static_cast<uint32_t> (NUM_NODES * P_NORMAL);
  const uint32_t numAdvanced = static_cast<uint32_t> (NUM_NODES * P_ADVANCED);
  const uint32_t numSuper = NUM_NODES - numNormal - numAdvanced;
  const double totalInitialEnergy = numSuper * E_SUP
                                  + numAdvanced * E_ADV
                                  + numNormal * E0;

  for (uint32_t i = 0; i < NUM_NODES; ++i)
    {
      SensorNode &n = g_nodes[i];
      n.id = i;
      n.x = RandUniform () * AREA_X;
      n.y = RandUniform () * AREA_Y;
      n.alive = true;
      n.isCH = false;
      n.isRelayCH = false;
      n.clusterId = -1;
      n.roundsSinceCH = 0;

      n.sensedValue = SENSED_INIT_MIN + RandUniform () * (SENSED_INIT_MAX - SENSED_INIT_MIN);
      n.lastSentValue = n.sensedValue;
      n.hasReportedOnce = false;
      n.deathRound = 0;
      n.everWasCH = false;
      n.everWasRelayCH = false;
      n.relayEnergyThisRound = 0.0;
      if (i < numSuper)
        {
          n.type = SUPER;
          n.initialEnergy = E_SUP;
        }
      else if (i < numSuper + numAdvanced)
        {
          n.type = ADVANCED;
          n.initialEnergy = E_ADV;
        }
      else
        {
          n.type = NORMAL;
          n.initialEnergy = E0;
        }
      n.energy = n.initialEnergy;
    }

  NS_LOG_UNCOND ("[Progress] DeployNodes: deployed " << NUM_NODES << " nodes ("
                 << numSuper << " super, " << numAdvanced << " advanced, "
                 << numNormal << " normal), total initial energy: "
                 << totalInitialEnergy << " J.");

  BuildDistanceCache ();
}

static double
NetworkTotalEnergy ()
{
  double sum = 0.0;
  for (auto &n : g_nodes)
    {
      if (n.alive)
        {
          sum += n.energy;
        }
    }
  return sum;
}

static uint32_t
CountAlive ()
{
  uint32_t c = 0;
  for (auto &n : g_nodes)
    {
      if (n.alive)
        {
          ++c;
        }
    }
  return c;
}

struct FitnessBreakdown
{
  double fp1, fp2, fp3, fp4, fp5, fp6, F;
};

static double
ComputeFP1 (const std::vector<uint32_t> &aliveIdx, double eTotal)
{
  double sum = 0.0;
  for (uint32_t idx : aliveIdx)
    {
      sum += g_nodes[idx].energy / eTotal;
    }
  if (sum <= 0.0)
    {
      return 1.0;
    }
  return 1.0 / sum;
}

static double
ComputeFP2 (const std::vector<uint32_t> &aliveIdx, double eTotal)
{
  double sum = 0.0;
  double safeETotal = (std::fabs (eTotal) < 1e-12) ? 1e-12 : eTotal;
  for (uint32_t idx : aliveIdx)
    {
      sum += (g_nodes[idx].initialEnergy / safeETotal);
    }
  if (std::fabs (sum) < 1e-12)
    {
      sum = 1e-12;
    }
  return 1.0 / sum;
}

static double
ComputeFP3 (const std::vector<uint32_t> &aliveIdx)
{
  return static_cast<double> (aliveIdx.size ());
}

static double
AvgDistAllNodesToSink (const std::vector<uint32_t> &aliveIdx)
{
  if (aliveIdx.empty ())
    {
      return 1.0;
    }
  double sum = 0.0;
  for (uint32_t idx : aliveIdx)
    {
      sum += DistToSink (g_nodes[idx]);
    }
  return sum / static_cast<double> (aliveIdx.size ());
}

static double
ComputeFP4 (const std::vector<uint32_t> &clusterMembers,
            const std::vector<uint32_t> &aliveIdx)
{
  uint32_t nCl = static_cast<uint32_t> (clusterMembers.size ());
  double distSum = 0.0;
  if (nCl > 1)
    {
      for (size_t i = 0; i < clusterMembers.size (); ++i)
        {
          for (size_t j = i + 1; j < clusterMembers.size (); ++j)
            {
              distSum += DistNodes (clusterMembers[i], clusterMembers[j]);
            }
        }
    }
  double firstTerm = 0.0;
  if (nCl > 0)
    {
      firstTerm = distSum / (static_cast<double> (nCl) * static_cast<double> (nCl));
    }
  double avgDistAll = AvgDistAllNodesToSink (aliveIdx);
  return firstTerm + avgDistAll;
}

static double
ComputeFP5 (const std::vector<uint32_t> &chSet, uint32_t nodeIdx)
{
  uint32_t nc = static_cast<uint32_t> (chSet.size ());
  if (nc == 0)
    {
      return 1.0;
    }
  double sumDistCh = 0.0;
  const SensorNode &self = g_nodes[nodeIdx];
  for (uint32_t chIdx : chSet)
    {
      if (chIdx == nodeIdx)
        {
          continue;
        }
      sumDistCh += DistNodes (nodeIdx, chIdx);
    }
  double avgDistCh = sumDistCh / static_cast<double> (nc);
  double distToSink = DistToSink (self);
  double fp5 = avgDistCh + distToSink;
  if (fp5 <= 0.0)
    {
      fp5 = 1e-9;
    }
  return fp5;
}

static uint32_t
GraphNeighborCount (uint32_t nodeIdx, const std::vector<uint32_t> &aliveIdx)
{
  uint32_t count = 0;
  for (uint32_t otherIdx : aliveIdx)
    {
      if (otherIdx != nodeIdx && DistNodes (nodeIdx, otherIdx) <= 30.0)
        {
          ++count;
        }
    }
  return count;
}

static void
BuildClusters (const std::vector<uint32_t> &aliveIdx,
               const std::vector<uint32_t> &chSet,
               std::vector<std::vector<uint32_t>> &clusters,
               std::vector<int> &assignment);

static double
ComputeFitness (const std::vector<uint32_t> &aliveIdx,
                const std::vector<uint32_t> &chSet,
                const std::vector<std::vector<uint32_t>> &clusters,
                double eTotal,
                FitnessBreakdown &fb);
static void
BeginGraphSageSnapshot (const std::vector<uint32_t> &aliveIdx, uint32_t round)
{
  if (!ENABLE_GRAPHSAGE_DATA || !GRAPHSAGE_DATA_MODE)
    {
      return;
    }

  g_pendingGraphSnapshot.clear ();
  g_pendingGraphSnapshot.reserve (aliveIdx.size ());
  for (uint32_t nodeIdx : aliveIdx)
    {
      GraphSageSnapshot snapshot;
      snapshot.seed = SIMULATION_SEED;
      snapshot.round = round;
      snapshot.targetNode = nodeIdx;
      snapshot.residualRatio = g_nodes[nodeIdx].energy /
                               std::max (g_nodes[nodeIdx].initialEnergy, 1e-12);
      snapshot.normalType = g_nodes[nodeIdx].type == NORMAL ? 1 : 0;
      snapshot.advancedType = g_nodes[nodeIdx].type == ADVANCED ? 1 : 0;
      snapshot.superType = g_nodes[nodeIdx].type == SUPER ? 1 : 0;
      snapshot.distanceToSink = DistToSink (g_nodes[nodeIdx]);
      snapshot.neighborCount = GraphNeighborCount (nodeIdx, aliveIdx);
      snapshot.localDensity = aliveIdx.empty () ? 0.0
                                                 : static_cast<double> (snapshot.neighborCount) /
                                                   static_cast<double> (aliveIdx.size ());
      snapshot.roundsSinceCHNormalized = std::min (1.0,
        static_cast<double> (g_nodes[nodeIdx].roundsSinceCH) / 10.0);
      snapshot.energyAtSelection = g_nodes[nodeIdx].energy;
      snapshot.sampleWeight = 1.0;
      snapshot.utilityEnergy = 0.0;
      snapshot.utilityCluster = 0.0;
      snapshot.utilityClusterEnergy = 0.0;
      snapshot.relayCost = 0.0;
      g_pendingGraphSnapshot.push_back (snapshot);

    }

  if (g_graphEdgesCsv.is_open ())
    {
      for (size_t i = 0; i < aliveIdx.size (); ++i)
        for (size_t j = i + 1; j < aliveIdx.size (); ++j)
          if (DistNodes (aliveIdx[i], aliveIdx[j]) <= 30.0)
            {
              g_graphEdgesCsv << SIMULATION_SEED << "," << round << ","
                              << aliveIdx[i] << "," << aliveIdx[j] << "\n"
                              << SIMULATION_SEED << "," << round << ","
                              << aliveIdx[j] << "," << aliveIdx[i] << "\n";
            }
    }
}

static void
CompleteGraphSageSnapshot (const std::vector<uint32_t> &chSet, double eTotal)
{
  if (g_pendingGraphSnapshot.empty () || chSet.empty ())
    {
      return;
    }
  std::vector<std::vector<uint32_t>> clusters;
  std::vector<int> assignment;
  std::vector<uint32_t> aliveIdx;
  for (auto &node : g_nodes) if (node.alive) aliveIdx.push_back (node.id);
  BuildClusters (aliveIdx, chSet, clusters, assignment);

  FitnessBreakdown fb;
  double fitness = ComputeFitness (aliveIdx, chSet, clusters, eTotal, fb);

  if (g_graphStateCsv.is_open ())
    {
      for (const GraphSageSnapshot &snapshot : g_pendingGraphSnapshot)
        g_graphStateCsv << snapshot.seed << "," << snapshot.round << ","
          << snapshot.targetNode << "," << snapshot.distanceToSink << ","
          << snapshot.residualRatio << "," << snapshot.normalType << ","
          << snapshot.advancedType << "," << snapshot.superType << ","
          << snapshot.roundsSinceCHNormalized << "," << snapshot.neighborCount << ","
          << snapshot.localDensity << "," << fitness << "\n";
    }

  for (GraphSageSnapshot &snapshot : g_pendingGraphSnapshot)
    {
      auto chPosition = std::find (chSet.begin (), chSet.end (), snapshot.targetNode);
      if (chPosition == chSet.end ()) continue;
      size_t position = static_cast<size_t> (chPosition - chSet.begin ());
      if (position < clusters.size ()) snapshot.cluster = clusters[position];
      for (uint32_t member : snapshot.cluster)
        snapshot.clusterEnergyAtSelection.push_back (g_nodes[member].energy);
      snapshot.fitness = fitness;
      g_pendingSamples.push_back (snapshot);
    }
  g_pendingGraphSnapshot.clear ();
}
static void
CompleteGraphSageSnapshot (const std::vector<uint32_t> &chSet)
{
  if (g_pendingGraphSnapshot.empty () || chSet.empty ())
    {
      return;
    }
  std::vector<std::vector<uint32_t>> clusters;
  std::vector<int> assignment;
  std::vector<uint32_t> aliveIdx;
  for (auto &node : g_nodes) if (node.alive) aliveIdx.push_back (node.id);
  BuildClusters (aliveIdx, chSet, clusters, assignment);
  for (GraphSageSnapshot &snapshot : g_pendingGraphSnapshot)
    {
      auto chPosition = std::find (chSet.begin (), chSet.end (), snapshot.targetNode);
      if (chPosition == chSet.end ()) continue;
      size_t position = static_cast<size_t> (chPosition - chSet.begin ());
      if (position < clusters.size ()) snapshot.cluster = clusters[position];
      for (uint32_t member : snapshot.cluster)
        snapshot.clusterEnergyAtSelection.push_back (g_nodes[member].energy);
      g_pendingSamples.push_back (snapshot);
    }
  g_pendingGraphSnapshot.clear ();
}

static void
CompletePendingGraphSageSamples (uint32_t round)
{
  if (!ENABLE_GRAPHSAGE_DATA || GRAPHSAGE_K == 0) return;
  for (auto it = g_pendingSamples.begin (); it != g_pendingSamples.end (); )
    {
      if (round < it->round + GRAPHSAGE_K) { ++it; continue; }
      bool valid = true;
      double endEnergy = g_nodes[it->targetNode].energy;
      double clusterStart = 0.0, clusterEnd = 0.0;
      uint32_t aliveMembers = 0;
      for (size_t memberPos = 0; memberPos < it->cluster.size (); ++memberPos)
        {
          uint32_t member = it->cluster[memberPos];
          clusterStart += it->clusterEnergyAtSelection[memberPos];
          clusterEnd += g_nodes[member].alive ? g_nodes[member].energy : 0.0;
          aliveMembers += g_nodes[member].alive ? 1 : 0;
        }
      if (clusterStart <= 0.0 ||
          (g_lastNodeDeadRound > 0 && it->round + GRAPHSAGE_K > g_lastNodeDeadRound))
        valid = false;
      if (valid)
        {
          it->utilityEnergy = endEnergy / std::max (it->energyAtSelection, 1e-12);
          it->utilityCluster = it->cluster.empty () ? 0.0
            : static_cast<double> (aliveMembers) / it->cluster.size ();
          it->utilityClusterEnergy = clusterEnd / clusterStart;
          if (g_graphNodesCsv.is_open ())
            {
              g_graphNodesCsv << it->seed << "," << it->round << ","
                << it->targetNode << "," << it->distanceToSink << ","
                << it->residualRatio << "," << it->normalType << ","
                << it->advancedType << "," << it->superType << ","
                << it->roundsSinceCHNormalized << "," << it->neighborCount << ","
                << it->localDensity << "," << it->utilityEnergy << ","
                << it->utilityCluster << "," << it->utilityClusterEnergy << ","
                << it->relayCost << "," << it->sampleWeight << ","
                << it->fitness << "\n";
            }
        }
      it = g_pendingSamples.erase (it);
    }
}

static void
AccumulatePendingRelayCosts (uint32_t round)
{
  for (GraphSageSnapshot &sample : g_pendingSamples)
    {
      if (round >= sample.round && round < sample.round + GRAPHSAGE_K &&
          sample.targetNode < g_nodes.size ())
        {
          sample.relayCost += g_nodes[sample.targetNode].relayEnergyThisRound;
        }
    }
}

static void
StartGraphSageInferenceServer ()
{
  if ((!ENABLE_GRAPHSAGE && !ENABLE_MLP) || g_graphSageSocket >= 0)
    {
      return;
    }

  g_graphSagePort = 39000 + static_cast<int> (SIMULATION_SEED % 1000);
  std::string command = GRAPHSAGE_PYTHON + " \"" + GRAPHSAGE_INFERENCE_SCRIPT +
                        "\" --server --port " + std::to_string (g_graphSagePort) +
                        " --mode " + (ENABLE_MLP ? "mlp" : "graphsage") +
                        " --model \"" + (ENABLE_MLP ? MLP_MODEL : GRAPHSAGE_MODEL) +
                        "\" >/dev/null 2>&1 &";
  std::system (command.c_str ());

  for (uint32_t attempt = 0; attempt < 100; ++attempt)
    {
      g_graphSageSocket = socket (AF_INET, SOCK_STREAM, 0);
      if (g_graphSageSocket < 0)
        {
          break;
        }
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_port = htons (static_cast<uint16_t> (g_graphSagePort));
      inet_pton (AF_INET, "127.0.0.1", &address.sin_addr);
      if (connect (g_graphSageSocket, reinterpret_cast<sockaddr *> (&address),
                   sizeof (address)) == 0)
        {
          return;
        }
      close (g_graphSageSocket);
      g_graphSageSocket = -1;
      std::this_thread::sleep_for (std::chrono::milliseconds (50));
    }
  NS_LOG_UNCOND ("[GraphSAGE] could not connect to inference server");
}

static void
StopGraphSageInferenceServer ()
{
  if (g_graphSageSocket >= 0)
    {
      const std::string quit = "QUIT\n";
      send (g_graphSageSocket, quit.c_str (), quit.size (), 0);
      close (g_graphSageSocket);
      g_graphSageSocket = -1;
    }
}

static void
RunGraphSageInference (const std::vector<uint32_t> &aliveIdx, uint32_t round)
{
  g_graphSageScores.assign (NUM_NODES, 0.0);
  if (!ENABLE_GRAPHSAGE && !ENABLE_MLP)
    {
      return;
    }
  StartGraphSageInferenceServer ();
  if (g_graphSageSocket < 0)
    {
      return;
    }

  const std::string inputNodes = OutputFile ("graphsage_live_nodes.csv");
  const std::string inputEdges = OutputFile ("graphsage_live_edges.csv");
  std::ofstream nodes (inputNodes);
    nodes << "seed,round,node_id,distance_to_sink,residual_ratio,normal_type,"
      << "advanced_type,super_type,rounds_since_ch,neighbor_count,local_density\n";
  for (uint32_t nodeIdx : aliveIdx)
    {
    uint32_t neighbors = GraphNeighborCount (nodeIdx, aliveIdx);
    nodes << SIMULATION_SEED << "," << round << "," << nodeIdx << ","
      << DistToSink (g_nodes[nodeIdx]) << ","
      << g_nodes[nodeIdx].energy / std::max (g_nodes[nodeIdx].initialEnergy, 1e-12) << ","
      << (g_nodes[nodeIdx].type == NORMAL ? 1 : 0) << ","
      << (g_nodes[nodeIdx].type == ADVANCED ? 1 : 0) << ","
      << (g_nodes[nodeIdx].type == SUPER ? 1 : 0) << ","
      << std::min (1.0, static_cast<double> (g_nodes[nodeIdx].roundsSinceCH) / 10.0) << ","
      << neighbors << ","
      << (aliveIdx.empty () ? 0.0 : static_cast<double> (neighbors) / aliveIdx.size ()) << "\n";
    }
  nodes.close ();

  std::ofstream edges (inputEdges);
  edges << "seed,round,source,destination\n";
  for (size_t i = 0; i < aliveIdx.size (); ++i)
    {
      for (size_t j = i + 1; j < aliveIdx.size (); ++j)
        {
          if (DistNodes (aliveIdx[i], aliveIdx[j]) <= 30.0)
            {
              edges << SIMULATION_SEED << "," << round << ","
                    << aliveIdx[i] << "," << aliveIdx[j] << "\n";
              edges << SIMULATION_SEED << "," << round << ","
                    << aliveIdx[j] << "," << aliveIdx[i] << "\n";
            }
        }
    }
  edges.close ();

  std::string request = inputNodes + "\t" + inputEdges + "\t" +
                        std::to_string (SIMULATION_SEED) + "\t" +
                        std::to_string (round) + "\n";
  if (send (g_graphSageSocket, request.c_str (), request.size (), 0) < 0)
    {
      NS_LOG_UNCOND ("[GraphSAGE] inference failed at round " << round);
      return;
    }

  std::string response;
  char buffer[4096];
  while (true)
    {
      ssize_t received = recv (g_graphSageSocket, buffer, sizeof (buffer), 0);
      if (received <= 0)
        {
          break;
        }
      response.append (buffer, static_cast<size_t> (received));
      size_t end = response.find ("END\n");
      if (end != std::string::npos)
        {
          response.resize (end);
          break;
        }
    }
  std::stringstream scoreStream (response);
  uint32_t nodeId;
  double score;
  char comma;
  while (scoreStream >> nodeId >> comma >> score)
    {
      if (nodeId < g_graphSageScores.size ())
        {
          g_graphSageScores[nodeId] = score;
          NS_LOG_UNCOND ("[GraphSAGE] seed=" << SIMULATION_SEED
                         << " round=" << round << " node=" << nodeId
                         << " score=" << score);
        }
    }
}

static double
ComputeFP6 (const std::vector<uint32_t> &aliveIdx)
{
  double sum = 0.0;
  for (uint32_t idx : aliveIdx)
    {
      double lb = std::min (0.1 * (static_cast<double> (g_nodes[idx].roundsSinceCH) + 1.0), 1.0);
      sum += lb;
    }
  if (sum <= 0.0)
    {
      sum = 1e-9;
    }
  return 1.0 / sum;
}

static double
ComputeFitness (const std::vector<uint32_t> &aliveIdx,
                 const std::vector<uint32_t> &chSet,
                 const std::vector<std::vector<uint32_t>> &clusters,
                 double eTotal,
                 FitnessBreakdown &fb)
{
  fb.fp1 = ComputeFP1 (aliveIdx, eTotal);
  fb.fp2 = ComputeFP2 (aliveIdx, eTotal);
  fb.fp3 = ComputeFP3 (aliveIdx);

  double fp4Sum = 0.0;
  if (!clusters.empty ())
    {
      for (auto &cl : clusters)
        {
          fp4Sum += ComputeFP4 (cl, aliveIdx);
        }
      fp4Sum /= static_cast<double> (clusters.size ());
    }
  fb.fp4 = fp4Sum;

  double fp5Sum = 0.0;
  if (!chSet.empty ())
    {
      for (uint32_t chIdx : chSet)
        {
          fp5Sum += ComputeFP5 (chSet, chIdx);
        }
      fp5Sum /= static_cast<double> (chSet.size ());
    }
  fb.fp5 = fp5Sum;

  fb.fp6 = ComputeFP6 (aliveIdx);

  double denom = W_FP1 * fb.fp1 + W_FP2 * fb.fp2 + W_FP3 * fb.fp3 +
                 W_FP4 * fb.fp4 + W_FP5 * fb.fp5 + W_FP6 * fb.fp6;
  if (std::fabs (denom) < 1e-12)
    {
      denom = 1e-12;
    }
  fb.F = 1.0 / denom;
  return fb.F;
}

struct Chromosome
{
  std::vector<uint8_t> bits;
  double fitness;
  double fp1, fp2, fp3, fp4, fp5, fp6;
  uint32_t rank;
  double crowdingDistance;
};

static std::vector<uint32_t>
GetAliveIndices ()
{
  std::vector<uint32_t> aliveIdx;
  for (auto &n : g_nodes)
    {
      if (n.alive)
        {
          aliveIdx.push_back (n.id);
        }
    }
  return aliveIdx;
}

static void
BuildClusters (const std::vector<uint32_t> &aliveIdx,
               const std::vector<uint32_t> &chSet,
               std::vector<std::vector<uint32_t>> &clusters,
               std::vector<int> &assignment                                     )
{
  clusters.clear ();
  clusters.resize (chSet.size ());
  assignment.assign (NUM_NODES, -1);

  if (chSet.empty ())
    {
      return;
    }

  for (uint32_t idx : aliveIdx)
    {
      bool isCh = std::find (chSet.begin (), chSet.end (), idx) != chSet.end ();
      if (isCh)
        {
          continue;
        }
      double bestDist = std::numeric_limits<double>::max ();
      int bestCh = -1;
      size_t bestChPos = 0;
      for (size_t c = 0; c < chSet.size (); ++c)
        {
          double d = DistNodes (idx, chSet[c]);
          if (d < bestDist)
            {
              bestDist = d;
              bestCh = static_cast<int> (chSet[c]);
              bestChPos = c;
            }
        }
      assignment[idx] = bestCh;
      clusters[bestChPos].push_back (idx);
    }

  for (size_t c = 0; c < chSet.size (); ++c)
    {
      clusters[c].push_back (chSet[c]);
    }
}

static std::vector<uint32_t>
ChromosomeToChSet (const Chromosome &c, const std::vector<uint32_t> &aliveIdx)
{
  std::vector<uint32_t> chSet;
  for (size_t i = 0; i < aliveIdx.size (); ++i)
    {
      if (c.bits[i] == 1)
        {
          chSet.push_back (aliveIdx[i]);
        }
    }
  return chSet;
}

static void
EvaluateChromosome (Chromosome &c, const std::vector<uint32_t> &aliveIdx, double eTotal)
{
  std::vector<uint32_t> chSet = ChromosomeToChSet (c, aliveIdx);
  if (chSet.empty ())
    {
      c.fitness = std::numeric_limits<double>::max ();
      c.fp1 = c.fp2 = c.fp3 = c.fp4 = c.fp5 = c.fp6 = 0.0;
      return;
    }
  std::vector<std::vector<uint32_t>> clusters;
  std::vector<int> assignment;
  BuildClusters (aliveIdx, chSet, clusters, assignment);

  FitnessBreakdown fb;
  c.fitness = ComputeFitness (aliveIdx, chSet, clusters, eTotal, fb);
  c.fp1 = fb.fp1;
  c.fp2 = fb.fp2;
  c.fp3 = fb.fp3;
  c.fp4 = fb.fp4;
  c.fp5 = fb.fp5;
  c.fp6 = fb.fp6;

  if ((ENABLE_GRAPHSAGE || ENABLE_MLP) && !chSet.empty ())
    {
      double graphScore = 0.0;
      for (uint32_t nodeIdx : chSet)
        {
          graphScore += g_graphSageScores[nodeIdx];
        }
      graphScore /= static_cast<double> (chSet.size ());
      double normalizedFitness = (c.fitness - GRAPH_FITNESS_MEAN) /
                 std::max (GRAPH_FITNESS_STD, 1e-12);
      double normalizedUtility = (graphScore - GRAPH_UTILITY_MEAN) /
                 std::max (GRAPH_UTILITY_STD, 1e-12);
      c.fitness = normalizedFitness - GRAPH_SAGE_WEIGHT * normalizedUtility;
    }
}

static void
AssignRankAndCrowding (std::vector<Chromosome> &pop)
{
  std::vector<size_t> order (pop.size ());
  for (size_t i = 0; i < order.size (); ++i) order[i] = i;
  std::sort (order.begin (), order.end (), [&pop] (size_t a, size_t b) {
    return pop[a].fitness < pop[b].fitness;
  });
  for (size_t r = 0; r < order.size (); ++r)
    {
      pop[order[r]].rank = static_cast<uint32_t> (r);
    }
  double minF = pop[order.front ()].fitness;
  double maxF = pop[order.back ()].fitness;
  double range = (maxF - minF) > 1e-12 ? (maxF - minF) : 1.0;
  for (auto &c : pop)
    {
      c.crowdingDistance = (maxF - c.fitness) / range;
    }
}

static const Chromosome &
BinaryTournamentSelect (const std::vector<Chromosome> &pop)
{
  size_t i = static_cast<size_t> (RandUniform () * pop.size ());
  size_t j = static_cast<size_t> (RandUniform () * pop.size ());
  if (i >= pop.size ()) i = pop.size () - 1;
  if (j >= pop.size ()) j = pop.size () - 1;
  const Chromosome &a = pop[i];
  const Chromosome &b = pop[j];
  bool aWins;
  if (a.rank != b.rank)
    {
      aWins = a.rank < b.rank;
    }
  else
    {
      aWins = a.crowdingDistance > b.crowdingDistance;
    }
  return aWins ? a : b;
}

static void
AdaptiveCrossover (const Chromosome &p1, const std::vector<Chromosome> &pop,
                    Chromosome &child1, Chromosome &child2)
{
  size_t bestIdx = 0;
  double bestDiff = -1.0;
  for (size_t k = 0; k < pop.size (); ++k)
    {
      double diff = 0.0;
      for (size_t g = 0; g < p1.bits.size (); ++g)
        {
          diff += std::fabs (static_cast<double> (p1.bits[g]) -
                              static_cast<double> (pop[k].bits[g]));
        }
      if (diff > bestDiff)
        {
          bestDiff = diff;
          bestIdx = k;
        }
    }
  const Chromosome &p2 = pop[bestIdx];

  double pc = PC_MIN + RandUniform () * (PC_MAX - PC_MIN);
  child1 = p1;
  child2 = p2;
  for (size_t g = 0; g < p1.bits.size (); ++g)
    {
      if (RandUniform () < pc)
        {
          child1.bits[g] = p2.bits[g];
          child2.bits[g] = p1.bits[g];
        }
    }
}

static void
MoveMutation (Chromosome &c1, Chromosome &c2)
{
  if (RandUniform () < PM && !c1.bits.empty ())
    {
      size_t pos = static_cast<size_t> (RandUniform () * c1.bits.size ());
      if (pos >= c1.bits.size ()) pos = c1.bits.size () - 1;
      if (c1.bits != c2.bits)
        {
          uint8_t val = c1.bits[pos];
          c2.bits[pos] = val;
        }
    }

  for (auto *child : {&c1, &c2})
    {
      for (uint8_t &bit : child->bits)
        {
          if (RandUniform () < PM)
            {
              bit = static_cast<uint8_t> (1 - bit);
            }
        }
    }
}

static std::vector<uint32_t>
RunNCOGA_GA (const std::vector<uint32_t> &aliveIdx, double eTotal,
             std::vector<double> &selectedFp5)
{
  uint32_t M = static_cast<uint32_t> (aliveIdx.size ());
  std::vector<Chromosome> population (POP_SIZE);

  double avgEnergy = eTotal / std::max<uint32_t> (M, 1);
  double thresholdEnergy = 0.5 * avgEnergy;

  for (auto &c : population)
    {
      c.bits.assign (M, 0);
      for (uint32_t j = 0; j < M; ++j)
        {
          const SensorNode &node = g_nodes[aliveIdx[j]];
          if (node.energy <= thresholdEnergy)
            {
              c.bits[j] = 0;
              continue;
            }
          double r = RandUniform ();
          c.bits[j] = (r <= CHP_INIT) ? 1 : 0;
        }
      EvaluateChromosome (c, aliveIdx, eTotal);
    }
  AssignRankAndCrowding (population);

  NS_LOG_UNCOND ("  [Progress] RunNCOGA_GA: initialized population of " << POP_SIZE
                 << " chromosomes over " << M << " alive nodes.");

  for (uint32_t gen = 0; gen < NUM_GENERATIONS; ++gen)
    {
      std::vector<Chromosome> newPop;
      newPop.reserve (POP_SIZE);
      while (newPop.size () < POP_SIZE)
        {
          const Chromosome &parent1 = BinaryTournamentSelect (population);
          Chromosome child1, child2;
          AdaptiveCrossover (parent1, population, child1, child2);
          MoveMutation (child1, child2);
          EvaluateChromosome (child1, aliveIdx, eTotal);
          EvaluateChromosome (child2, aliveIdx, eTotal);
          newPop.push_back (child1);
          if (newPop.size () < POP_SIZE)
            {
              newPop.push_back (child2);
            }
        }

      AssignRankAndCrowding (population);
      size_t bestPrevIdx = 0;
      for (size_t i = 1; i < population.size (); ++i)
        {
          if (population[i].fitness < population[bestPrevIdx].fitness)
            {
              bestPrevIdx = i;
            }
        }
      AssignRankAndCrowding (newPop);
      size_t worstNewIdx = 0;
      for (size_t i = 1; i < newPop.size (); ++i)
        {
          if (newPop[i].fitness > newPop[worstNewIdx].fitness)
            {
              worstNewIdx = i;
            }
        }
      if (population[bestPrevIdx].fitness < newPop[worstNewIdx].fitness)
        {
          newPop[worstNewIdx] = population[bestPrevIdx];
        }

      population = newPop;
      AssignRankAndCrowding (population);

    }

  size_t bestIdx = 0;
  for (size_t i = 1; i < population.size (); ++i)
    {
      if (population[i].fitness < population[bestIdx].fitness)
        {
          bestIdx = i;
        }
    }
  std::vector<uint32_t> selectedChSet =
    ChromosomeToChSet (population[bestIdx], aliveIdx);
  if (selectedChSet.empty () && !aliveIdx.empty ())
    {
      uint32_t bestEnergyIdx = aliveIdx.front ();
      for (uint32_t idx : aliveIdx)
        {
          if (g_nodes[idx].energy > g_nodes[bestEnergyIdx].energy)
            {
              bestEnergyIdx = idx;
            }
        }
      selectedChSet.push_back (bestEnergyIdx);
    }
  selectedFp5.clear ();
  selectedFp5.reserve (selectedChSet.size ());
  for (uint32_t chIdx : selectedChSet)
    {
      selectedFp5.push_back (ComputeFP5 (selectedChSet, chIdx));
    }
  return selectedChSet;
}

static void
RunSteadyStateRound (const std::vector<uint32_t> &chSet,
                      const std::vector<uint32_t> &aliveIdx,
                      const std::vector<double> &chFp5)
{
  for (auto &node : g_nodes)
    {
      node.relayEnergyThisRound = 0.0;
    }
  std::vector<std::vector<uint32_t>> clusters;
  std::vector<int> assignment;
  BuildClusters (aliveIdx, chSet, clusters, assignment);
  (void) clusters;

  double sumDistToSink = 0.0;
  uint32_t aliveCount = 0;
  for (uint32_t idx : aliveIdx)
    {
      if (g_nodes[idx].alive)
        {
          sumDistToSink += DistToSink (g_nodes[idx]);
          ++aliveCount;
        }
    }
  double avgDistToSink = aliveCount > 0
                           ? sumDistToSink / static_cast<double> (aliveCount)
                           : 0.0;

  std::vector<int> relayTarget (NUM_NODES, -1);
  for (auto &node : g_nodes)
    {
      node.isRelayCH = false;
    }
  for (size_t c = 0; c < chSet.size (); ++c)
    {
      uint32_t chIdx = chSet[c];
      if (!g_nodes[chIdx].alive || DistToSink (g_nodes[chIdx]) <= avgDistToSink)
        {
          continue;
        }

      double sourceDistToSink = DistToSink (g_nodes[chIdx]);
      int bestRelay = -1;
      double bestCmd = -std::numeric_limits<double>::max ();
      for (size_t candidatePos = 0; candidatePos < chSet.size (); ++candidatePos)
        {
          uint32_t candidateIdx = chSet[candidatePos];
          if (candidateIdx == chIdx || !g_nodes[candidateIdx].alive ||
              candidatePos >= chFp5.size () ||
              DistToSink (g_nodes[candidateIdx]) > avgDistToSink ||
              DistToSink (g_nodes[candidateIdx]) >= sourceDistToSink)
            {
              continue;
            }

          double cmd = 1.0 / chFp5[candidatePos];
          if (cmd > bestCmd)
            {
              bestCmd = cmd;
              bestRelay = static_cast<int> (candidateIdx);
            }
        }

      if (bestRelay >= 0)
        {
          relayTarget[chIdx] = bestRelay;
          g_nodes[static_cast<uint32_t> (bestRelay)].isRelayCH = true;
          g_nodes[static_cast<uint32_t> (bestRelay)].everWasRelayCH = true;
        }
    }

  static std::normal_distribution<double> driftDist (0.0, SENSED_DRIFT_STD);

  std::vector<uint32_t> receivedPackets (NUM_NODES, 0);

  auto thresholdReportDue = [&] (SensorNode &node) {
    node.sensedValue += driftDist (g_rng);
    node.sensedValue = std::max (0.0, std::min (1.0, node.sensedValue));

    if (!node.hasReportedOnce)
      {
        if (node.sensedValue < HARD_THRESHOLD)
          {
            return false;
          }
        node.hasReportedOnce = true;
        return true;
      }
    return std::fabs (node.sensedValue - node.lastSentValue) >= SOFT_THRESHOLD;
  };

  for (uint32_t idx : aliveIdx)
    {
      bool isCh = std::find (chSet.begin (), chSet.end (), idx) != chSet.end ();
      if (isCh)
        {
          size_t chPos = static_cast<size_t> (
            std::find (chSet.begin (), chSet.end (), idx) - chSet.begin ());
          if (chPos >= clusters.size () || clusters[chPos].size () > 1)
            {
              continue;
            }
        }

      SensorNode &member = g_nodes[idx];

      if (!isCh && assignment[idx] < 0)
        {
          if (thresholdReportDue (member))
            {
              member.lastSentValue = member.sensedValue;
              double txCost = Etx (PACKET_SIZE, DistToSink (member));
              if (member.energy > txCost)
                {
                  member.energy -= txCost;
                  g_totalPacketsToSink += 1;
                }
              else
                {
                  member.energy = 0.0;
                  member.alive = false;
                  member.deathRound = g_round;
                }
            }
          continue;
        }

      if (isCh)
        {
          if (thresholdReportDue (member))
            {
              member.lastSentValue = member.sensedValue;
              double txCost = Etx (PACKET_SIZE, DistToSink (member));
              if (member.energy > txCost)
                {
                  member.energy -= txCost;
                  g_totalPacketsToSink += 1;
                }
              else
                {
                  member.energy = 0.0;
                  member.alive = false;
                  member.deathRound = g_round;
                }
            }
          continue;
        }

      bool shouldSend = thresholdReportDue (member);
      if (!shouldSend)
        {
          continue;
        }

      member.lastSentValue = member.sensedValue;

      SensorNode &ch = g_nodes[assignment[idx]];
      double d = DistNodes (idx, static_cast<uint32_t> (assignment[idx]));
      double txCost = Etx (PACKET_SIZE, d);
      double rxCost = Erx (PACKET_SIZE);
      if (member.energy > txCost)
        {
          member.energy -= txCost;
        }
      else
        {
          member.energy = 0.0;
          member.alive = false;
          member.deathRound = g_round;
        }
      if (ch.alive)
        {
          if (ch.energy > rxCost)
            {
              ch.energy -= rxCost;

              receivedPackets[static_cast<uint32_t> (assignment[idx])] += 1;
            }
          else
            {
              ch.energy = 0.0;
              ch.alive = false;
              ch.deathRound = g_round;
            }
        }
    }

  std::vector<uint32_t> incomingRelay (NUM_NODES, 0);

  std::vector<uint32_t> chOrder (chSet);
  std::sort (chOrder.begin (), chOrder.end (),
             [] (uint32_t a, uint32_t b) {
               return DistToSink (g_nodes[a]) > DistToSink (g_nodes[b]);
             });

  for (uint32_t chIdx : chOrder)
    {
      SensorNode &ch = g_nodes[chIdx];
      if (!ch.alive)
        {

          continue;
        }

      uint32_t numReceived = receivedPackets[chIdx];
      uint32_t ownPacket = 0;
      if (numReceived > 0)
        {
          double aggCost = Edx (PACKET_SIZE, static_cast<double> (numReceived));
          if (ch.energy > aggCost)
            {
              ch.energy -= aggCost;
              ownPacket = 1;
            }
          else
            {
              ch.energy = 0.0;
              ch.alive = false;
              ch.deathRound = g_round;
              continue;
            }
        }

      uint32_t inflow = incomingRelay[chIdx];
      if (inflow > 0)
        {
          double rxCost = Erx (PACKET_SIZE) * static_cast<double> (inflow);
          ch.relayEnergyThisRound += rxCost;
          if (ch.energy > rxCost)
            {
              ch.energy -= rxCost;
            }
          else
            {
              ch.energy = 0.0;
              ch.alive = false;
              ch.deathRound = g_round;
              continue;
            }
        }

      uint32_t totalOutgoing = ownPacket + inflow;
      if (totalOutgoing == 0)
        {
          continue;
        }

      int nextHop = relayTarget[chIdx];
      double distTarget = DistToSink (ch);
      if (nextHop >= 0)
        {
          distTarget = DistNodes (chIdx, static_cast<uint32_t> (nextHop));
        }

      double txCost = Etx (PACKET_SIZE, distTarget);
      if (ch.energy > txCost)
        {
          ch.energy -= txCost;
          if (inflow > 0)
            {
              ch.relayEnergyThisRound += txCost;
            }
          if (nextHop < 0)
            {
              g_totalPacketsToSink += totalOutgoing;
            }
          else
            {
              incomingRelay[static_cast<uint32_t> (nextHop)] += totalOutgoing;
            }
        }
      else
        {
          ch.energy = 0.0;
          ch.alive = false;
          ch.deathRound = g_round;

        }
    }
}

static void
RunRound ()
{
  uint32_t alive = CountAlive ();
  if (alive == 0)
    {
      NS_LOG_UNCOND ("All nodes dead at round " << g_round << ". Simulation ending.");
      Simulator::Stop ();
      return;
    }

  std::vector<uint32_t> aliveIdx = GetAliveIndices ();
  double eTotal = NetworkTotalEnergy ();

  CompletePendingGraphSageSamples (g_round);

  NS_LOG_UNCOND ("[Progress] Round " << g_round << " starting -- alive nodes: "
                 << aliveIdx.size () << ", network energy: " << eTotal << " J");

  BeginGraphSageSnapshot (aliveIdx, g_round);
  RunGraphSageInference (aliveIdx, g_round);
  std::vector<double> chFp5;
  std::vector<uint32_t> chSet = RunNCOGA_GA (aliveIdx, eTotal, chFp5);

  if (GRAPHSAGE_DATA_MODE && SIMULATION_SEED <= 12 &&
      EXPLORATION_FRACTION > 0.0 && RandUniform () < EXPLORATION_FRACTION)
    {
      uint32_t added = 0;
      for (uint32_t idx : aliveIdx)
        {
          if (std::find (chSet.begin (), chSet.end (), idx) == chSet.end ())
            {
              chSet.push_back (idx);
              ++added;
              if (added == 2) break;
            }
        }
      for (auto &sample : g_pendingGraphSnapshot)
        {
          if (std::find (chSet.begin (), chSet.end (), sample.targetNode) != chSet.end ())
            sample.sampleWeight = 0.5;
        }
    }
  CompleteGraphSageSnapshot (chSet, eTotal);
  {
    std::ostringstream chOss;
    chOss << "[Progress] Round " << g_round << ": Cluster Heads = { ";
    for (size_t i = 0; i < chSet.size (); ++i)
      {
        chOss << chSet[i];
        if (i + 1 < chSet.size ())
          {
            chOss << ", ";
          }
      }
    chOss << " }";
    NS_LOG_UNCOND (chOss.str ());
  }
  NS_LOG_UNCOND ("[Progress] Round " << g_round << ": GA selected " << chSet.size ()
                 << " cluster head(s).");

  for (uint32_t idx : aliveIdx)
    {
      bool isCh = std::find (chSet.begin (), chSet.end (), idx) != chSet.end ();
      if (isCh)
        {
          g_nodes[idx].roundsSinceCH = 0;
          g_nodes[idx].isCH = true;
          g_nodes[idx].everWasCH = true;
        }
      else
        {
          g_nodes[idx].roundsSinceCH++;
          g_nodes[idx].isCH = false;
        }
    }

  RunSteadyStateRound (chSet, aliveIdx, chFp5);
  AccumulatePendingRelayCosts (g_round);

  uint32_t aliveAfter = CountAlive ();
  uint32_t deadAfter = NUM_NODES - aliveAfter;
  double remainingEnergy = NetworkTotalEnergy ();

  NS_LOG_UNCOND ("[Progress] Round " << g_round << " complete -- alive: " << aliveAfter
                 << ", dead: " << deadAfter << ", remaining energy: " << remainingEnergy
                 << " J, cumulative throughput: " << g_totalPacketsToSink << " packets");

  if (!g_fndRecorded && deadAfter >= 1)
    {
      g_firstNodeDeadRound = g_round;
      g_fndRecorded = true;
      NS_LOG_UNCOND ("First Node Dead (FND) at round " << g_round);
    }
  if (!g_hndRecorded && deadAfter >= NUM_NODES / 2)
    {
      g_halfNodesDeadRound = g_round;
      g_hndRecorded = true;
      NS_LOG_UNCOND ("Half Nodes Dead (HND) at round " << g_round);
    }
  if (!g_lndRecorded && aliveAfter == 0)
    {
      g_lastNodeDeadRound = g_round;
      g_lndRecorded = true;
      NS_LOG_UNCOND ("Last Node Dead (LND) / Network Lifetime at round " << g_round);
    }

  g_csv << g_round << "," << aliveAfter << "," << deadAfter << ","
        << remainingEnergy << "," << g_totalPacketsToSink << "," << chSet.size ()
        << "\n";
  g_sumChCount += static_cast<double> (chSet.size ());
  ++g_roundsExecuted;

  g_aliveCsv << g_round << "," << aliveAfter << "\n";

  g_round++;

  if (g_round < MAX_ROUNDS && aliveAfter > 0)
    {
      Simulator::Schedule (Seconds (1.0), &RunRound);
    }
  else
    {
      Simulator::Stop ();
    }
}

int
main (int argc, char *argv[])
{
  CommandLine cmd;
  cmd.AddValue ("seed", "Deterministic simulation seed", SIMULATION_SEED);
  cmd.AddValue ("enableGraphSAGE", "Use frozen GraphSAGE scores in GA", ENABLE_GRAPHSAGE);
  cmd.AddValue ("enableMLP", "Use frozen MLP scores in GA", ENABLE_MLP);
  cmd.AddValue ("enableGraphSAGEData", "Export seed-1 GraphSAGE training data",
                ENABLE_GRAPHSAGE_DATA);
  cmd.AddValue ("graphSageTrainingMode", "Mark this run as seed-1 training",
                GRAPHSAGE_TRAINING_MODE);
  cmd.AddValue ("graphSageModel", "Frozen GraphSAGE model path", GRAPHSAGE_MODEL);
  cmd.AddValue ("mlpModel", "Frozen MLP model path", MLP_MODEL);
  cmd.AddValue ("graphSageInferenceScript", "GraphSAGE inference script path",
                GRAPHSAGE_INFERENCE_SCRIPT);
  cmd.AddValue ("graphSagePython", "Python interpreter for GraphSAGE inference",
                GRAPHSAGE_PYTHON);
  cmd.AddValue ("graphSageK", "Forward utility horizon in rounds", GRAPHSAGE_K);
  cmd.AddValue ("explorationFraction", "Training-only exploration fraction",
                EXPLORATION_FRACTION);
  cmd.AddValue ("graphSageDataMode", "Enable utility sample collection",
                GRAPHSAGE_DATA_MODE);
  cmd.AddValue ("graphFitnessMean", "Frozen NCOGA fitness mean", GRAPH_FITNESS_MEAN);
  cmd.AddValue ("graphFitnessStd", "Frozen NCOGA fitness standard deviation", GRAPH_FITNESS_STD);
  cmd.AddValue ("graphUtilityMean", "Frozen utility mean", GRAPH_UTILITY_MEAN);
  cmd.AddValue ("graphUtilityStd", "Frozen utility standard deviation", GRAPH_UTILITY_STD);
  cmd.AddValue ("graphSageWeight", "Frozen hybrid fitness lambda", GRAPH_SAGE_WEIGHT);
  cmd.AddValue ("outputPrefix", "Prefix for this run's output files", OUTPUT_PREFIX);
  cmd.Parse (argc, argv);

  g_rng.seed (SIMULATION_SEED);
  g_graphSageScores.assign (NUM_NODES, 0.0);

  NS_LOG_UNCOND ("GraphSAGE training seed: 1");
  NS_LOG_UNCOND ("GraphSAGE training rounds: " << MAX_ROUNDS);
  NS_LOG_UNCOND ("Evaluation MAX_ROUNDS: " << MAX_ROUNDS);
  NS_LOG_UNCOND ("Current simulation seed: " << SIMULATION_SEED
                 << ", GraphSAGE enabled: " << (ENABLE_GRAPHSAGE ? "true" : "false"));

  LogComponentEnable ("NCOGA", LOG_LEVEL_INFO);

  g_resultsTxt.open (OutputFile ("ncoga_full_output.txt"));
  g_origClogBuf = std::clog.rdbuf ();
  g_teeBuf = new TeeStreamBuf (g_origClogBuf, g_resultsTxt.rdbuf ());
  std::clog.rdbuf (g_teeBuf);
  DeployNodes ();

  g_csv.open (OutputFile ("ncoga_results.csv"));
  g_csv << "round,alive_nodes,dead_nodes,remaining_energy,cumulative_throughput,num_ch\n";

  g_aliveCsv.open (OutputFile ("alive_nodes_vs_rounds.csv"));
  g_aliveCsv << "round,alive_nodes\n";

  g_aliveCsv << "0," << NUM_NODES << "\n";

  if (ENABLE_GRAPHSAGE_DATA && GRAPHSAGE_DATA_MODE && SIMULATION_SEED <= 16)
    {
      g_graphNodesCsv.open (OutputFile ("graphsage_training_nodes.csv"));
      g_graphNodesCsv << "seed,round,node_id,distance_to_sink,residual_ratio,"
              << "normal_type,advanced_type,super_type,rounds_since_ch,"
                      << "neighbor_count,local_density,U_E,U_C,U_CE,C_relay,sample_weight,F\n";
          g_graphStateCsv.open (OutputFile ("graphsage_training_graph_nodes.csv"));
          g_graphStateCsv << "seed,round,node_id,distance_to_sink,residual_ratio,"
                  << "normal_type,advanced_type,super_type,rounds_since_ch,"
                  << "neighbor_count,local_density,F\n";
      g_graphEdgesCsv.open (OutputFile ("graphsage_training_edges.csv"));
      g_graphEdgesCsv << "seed,round,source,destination\n";
    }
  NS_LOG_UNCOND ("Starting NCOGA simulation: N=" << NUM_NODES
                 << " area=" << AREA_X << "x" << AREA_Y);

  Simulator::Schedule (Seconds (0.0), &RunRound);
  Simulator::Run ();

  g_csv.close ();
  g_aliveCsv.close ();
  StopGraphSageInferenceServer ();
  if (g_graphNodesCsv.is_open ()) g_graphNodesCsv.close ();
  if (g_graphStateCsv.is_open ()) g_graphStateCsv.close ();
  if (g_graphEdgesCsv.is_open ()) g_graphEdgesCsv.close ();

  g_summaryCsv.open (OutputFile ("summary.csv"));
  g_summaryCsv << "seed,fnd,hnd,lnd,total_throughput,final_energy,avg_ch,num_rounds\n";
  g_summaryCsv << SIMULATION_SEED << ","
               << (g_firstNodeDeadRound > 0 ? static_cast<int> (g_firstNodeDeadRound) : -1)
               << ","
               << (g_halfNodesDeadRound > 0 ? static_cast<int> (g_halfNodesDeadRound) : -1)
               << ","
               << (g_lastNodeDeadRound > 0 ? static_cast<int> (g_lastNodeDeadRound) : -1)
               << "," << g_totalPacketsToSink << "," << NetworkTotalEnergy () << ","
               << (g_roundsExecuted > 0 ? g_sumChCount / g_roundsExecuted : 0.0)
               << "," << g_roundsExecuted << "\n";
  g_summaryCsv.close ();

  NS_LOG_UNCOND ("=== NCOGA Simulation Summary ===");
  NS_LOG_UNCOND ("First Node Dead (Stability Period): " << g_firstNodeDeadRound);
  NS_LOG_UNCOND ("Half Nodes Dead: " << g_halfNodesDeadRound);
  NS_LOG_UNCOND ("Last Node Dead (Network Lifetime): " << g_lastNodeDeadRound);
  NS_LOG_UNCOND ("Total packets delivered to sink (Throughput): " << g_totalPacketsToSink);

  NS_LOG_UNCOND ("Alive-nodes-vs-rounds series written to alive_nodes_vs_rounds.csv");

  if (g_lastNodeDeadRound > 0)
    {
      double fndRatio = static_cast<double> (g_firstNodeDeadRound) / g_lastNodeDeadRound;
      double hndRatio = static_cast<double> (g_halfNodesDeadRound) / g_lastNodeDeadRound;
      NS_LOG_UNCOND ("=== FND:HND:LND Ratio Check ===");
      NS_LOG_UNCOND ("FND / LND (stability period fraction) : " << fndRatio);
      NS_LOG_UNCOND ("HND / LND                              : " << hndRatio);
      NS_LOG_UNCOND ("FND : HND : LND                        : "
                     << g_firstNodeDeadRound << " : "
                     << g_halfNodesDeadRound << " : "
                     << g_lastNodeDeadRound);
    }
  else
    {
      NS_LOG_UNCOND ("=== FND:HND:LND Ratio Check ===");
      NS_LOG_UNCOND ("LND not reached within MAX_ROUNDS -- cannot compute ratios. "
                      "Consider raising MAX_ROUNDS or reviewing SOFT_THRESHOLD.");
    }

  NS_LOG_UNCOND ("Results written to ncoga_results.csv");

  {
    double sumNormal = 0.0, sumAdvanced = 0.0, sumSuper = 0.0;
    uint32_t cntNormal = 0, cntAdvanced = 0, cntSuper = 0;
    double sumRelay = 0.0, sumNonRelay = 0.0;
    uint32_t cntRelay = 0, cntNonRelay = 0;
    double sumEverCH = 0.0, sumNeverCH = 0.0;
    uint32_t cntEverCH = 0, cntNeverCH = 0;

    for (auto &n : g_nodes)
      {

        uint32_t effectiveDeath = n.alive ? g_round : n.deathRound;

        if (n.type == NORMAL) { sumNormal += effectiveDeath; cntNormal++; }
        else if (n.type == ADVANCED) { sumAdvanced += effectiveDeath; cntAdvanced++; }
        else { sumSuper += effectiveDeath; cntSuper++; }

        if (n.everWasRelayCH) { sumRelay += effectiveDeath; cntRelay++; }
        else { sumNonRelay += effectiveDeath; cntNonRelay++; }

        if (n.everWasCH) { sumEverCH += effectiveDeath; cntEverCH++; }
        else { sumNeverCH += effectiveDeath; cntNeverCH++; }
      }

    NS_LOG_UNCOND ("=== Death Round Breakdown by Tier ===");
    NS_LOG_UNCOND ("Normal   (n=" << cntNormal   << ") avg death round: "
                   << (cntNormal   ? sumNormal   / cntNormal   : 0.0));
    NS_LOG_UNCOND ("Advanced (n=" << cntAdvanced << ") avg death round: "
                   << (cntAdvanced ? sumAdvanced / cntAdvanced : 0.0));
    NS_LOG_UNCOND ("Super    (n=" << cntSuper    << ") avg death round: "
                   << (cntSuper    ? sumSuper    / cntSuper    : 0.0));

    NS_LOG_UNCOND ("=== Death Round Breakdown by Relay-CH Role ===");
    NS_LOG_UNCOND ("Ever relay CH     (n=" << cntRelay    << ") avg death round: "
                   << (cntRelay    ? sumRelay    / cntRelay    : 0.0));
    NS_LOG_UNCOND ("Never relay CH    (n=" << cntNonRelay << ") avg death round: "
                   << (cntNonRelay ? sumNonRelay / cntNonRelay : 0.0));

    NS_LOG_UNCOND ("=== Death Round Breakdown by CH Duty ===");
    NS_LOG_UNCOND ("Ever was CH       (n=" << cntEverCH   << ") avg death round: "
                   << (cntEverCH   ? sumEverCH   / cntEverCH   : 0.0));
    NS_LOG_UNCOND ("Never was CH      (n=" << cntNeverCH  << ") avg death round: "
                   << (cntNeverCH  ? sumNeverCH  / cntNeverCH  : 0.0));
  }
  Simulator::Destroy ();

  std::clog.rdbuf (g_origClogBuf);
  delete g_teeBuf;
  g_resultsTxt.close ();
  return 0;
}
