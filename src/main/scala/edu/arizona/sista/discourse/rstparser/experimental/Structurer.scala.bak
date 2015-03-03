package edu.arizona.sista.discourse.rstparser.experimental

import scala.math.{pow, sqrt}
import scala.util.Random.shuffle
import scala.concurrent.{Await, Future}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import edu.arizona.sista.processors.Document
import edu.arizona.sista.discourse.rstparser._
import edu.arizona.sista.discourse.rstparser.Utils.mkGoldEDUs
import breeze.linalg.SparseVector

class Structurer(val epochs: Int, val learningRate: Double, val relModel: RelationClassifier) {
  require(epochs > 0, "'epochs' should be greater than 0")
  require(learningRate >= 0 && learningRate <= 1, "'learningRate' should be between 0 and 1")

  val policy = new InterpolatedPolicy(relModel)
  val featureExtractor = new FeatureExtractor
  var avgWeights: SparseVector[Double] = _

  def train(treedocs: IndexedSeq[(DiscourseTree, Document)], corpusStats: CorpusStats, numShards: Int): LearnedPolicy = {
    avgWeights = SparseVector.zeros[Double](Integer.MAX_VALUE)

    def oneEpoch(indices: Seq[Int]): Future[SparseVector[Double]] = Future {
      var weights = avgWeights.copy
      var avgW = avgWeights.copy
      for (i <- indices) {
        val (tree, doc) = treedocs(i)
        val edus = mkGoldEDUs(tree, doc)
        val path = policy.getCompletePath(tree, doc)
        for (state <- path.init) {  // last state in path is a solution
          val nextStatesWithMergedIndex = getNextStatesWithMergedIndex(state, doc, edus, relModel)
          val (nextStates, mergedIndex) = nextStatesWithMergedIndex.unzip
          val costs = getStatesCosts(nextStates, tree)
          val (bestNextState, bestNextCost) = getCheapestState(nextStates, costs)
          val bestMerge = nextStatesWithMergedIndex.find(_._1 == bestNextState).get._2

          val predMerge = predictMerge(weights, state, doc, edus, corpusStats)
          val predNextState = nextStatesWithMergedIndex.find(_._2 == predMerge).get._1

          val predNextCost = StructureScorer.loss(tree, predNextState)

          if (predNextCost - bestNextCost > 0) {
            val bestFeatures = getFeatures(state, bestMerge, doc, edus, corpusStats)
            val predFeatures = getFeatures(state, predMerge, doc, edus, corpusStats)
            weights += bestFeatures
            weights -= predFeatures
          }

          avgW += weights
        }
      }
      avgW
    }

    for (i <- 0 until epochs) {
      println(s"epoch ${i+1} of $epochs")
      println(s"processing ${treedocs.size} documents")
      policy.expertProbability = pow(1 - learningRate, i)
      val indices = shuffle(Seq.range(0, treedocs.size))
      val shards = splitShards(indices, numShards)

      val futureWeights = Future.sequence(shards map oneEpoch)
      val allWeights = Await.result(futureWeights, Duration.Inf)

      for (w <- allWeights) {
        avgWeights += w
      }

      policy.learned = new LearnedPolicy(avgWeights, corpusStats, relModel)
    }

    policy.learned
  }

  def splitShards(indices: Seq[Int], numShards: Int): Seq[Seq[Int]] = {
    var remaining = indices
    val size = indices.size
    val shardSize = size / numShards
    val extraItems = size % numShards
    for (i <- 1 to numShards) yield {
      if (i == numShards) remaining
      else {
        val index = shardSize + (if (i <= extraItems) 1 else 0)
        val (shard, rest) = remaining splitAt index
        remaining = rest
        shard
      }
    }
  }

  def getFeatures(state: State,
                  merge: Int,
                  doc: Document,
                  edus: Array[Array[(Int, Int)]],
                  corpusStats: CorpusStats): SparseVector[Double] = {
    val left = state(merge)
    val right = state(merge + 1)
    val d = relModel.mkDatum(left, right, doc, edus, StructureClassifier.NEG)
    val ld = relModel.classOf(d)
    val (label, dir) = relModel.parseLabel(ld)
    featureExtractor.getFeatures(left, right, doc, edus, corpusStats, label)
  }

  def predictMerge(weights: SparseVector[Double],
                   state: State,
                   doc: Document,
                   edus: Array[Array[(Int, Int)]],
                   corpusStats: CorpusStats): Int = {
    val indicesWithScores = for (i <- 0 to state.size - 2) yield {
      val features = getFeatures(state, i, doc, edus, corpusStats)
      (i, (weights dot features))
    }
    // return the index with the maximum score
    indicesWithScores.maxBy(_._2)._1
  }
}
