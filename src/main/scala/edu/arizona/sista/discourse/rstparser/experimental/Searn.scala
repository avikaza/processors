package edu.arizona.sista.discourse.rstparser.experimental

import scala.util.Random
import scala.math.{ pow, sqrt }
import edu.arizona.sista.discourse.rstparser._
import edu.arizona.sista.discourse.rstparser.Utils.mkGoldEDUs
import edu.arizona.sista.discourse.rstparser.StructureClassifier.{POS, NEG}
import edu.arizona.sista.learning._
import edu.arizona.sista.processors.Document
import breeze.linalg._

class Searn(
    val epochs: Int,
    val learningRate: Double,
    val relModel: RelationClassifier
) {
  require(epochs > 0, "'epochs' should be greater than zero")
  require(learningRate >= 0 && learningRate <= 1, "'learningRate' should be between 0 and 1")

  val policy = new InterpolatedPolicy(relModel)
  val featureExtractor = new RelationFeatureExtractor

  def train(
    treedocs: IndexedSeq[(DiscourseTree, Document)],
    corpusStats: CorpusStats,
    scaleRange: ScaleRange[String]
  ): Unit = {
    // dataset indices
    val indices = Seq.range(0, treedocs.size)

    val classifier = new CostSensitiveClassifier[String, String](5, 1, Some(scaleRange))
    policy.learned = new LearnedPolicy(classifier, corpusStats, relModel)

    val protoDataset = new RVFDataset[String, String]
    protoDataset.labelLexicon.add(POS)
    protoDataset.labelLexicon.add(NEG)

    for (epoch <- 0 until epochs) {
      println(s"Searn epoch $epoch")
      // update the probability of using the expert policy
      // policy.expertProbability = pow(1 - learningRate, epoch)

      // searn builds a new dataset for each epoch
      val dataset = protoDataset.emptyDataset
      var costMatrix: DenseMatrix[Double] = null

      for (i <- Random.shuffle(indices)) {
        val (tree, doc) = treedocs(i)
        val edus = mkGoldEDUs(tree, doc)
        val path = policy.getCompletePath(tree, doc)

        for (state <- path.init) { // last state in path is a solution
          val nextStatesMergedIndex = getNextStatesWithMergedIndex(state, doc, edus, relModel)
          val (nextStates, mergedIndex) = nextStatesMergedIndex.unzip
          val costs = getStatesCosts(nextStates, tree)

          for ((i, c) <- mergedIndex zip costs) {
            val feats = policy.learned.getFeatures(state, i, doc, edus, corpusStats)
            val costVector = DenseVector(Array(
              dataset.labelLexicon.get(POS).get -> c,
              dataset.labelLexicon.get(NEG).get -> 0.0   // FIXME is this always zero?
            ).sortBy(_._1).map(_._2))

            dataset += new RVFDatum(NEG, feats)
            if (costMatrix == null) {
              costMatrix = costVector.toDenseMatrix
            } else {
              costMatrix = DenseMatrix.vertcat(costMatrix, costVector.toDenseMatrix)
            }
          }

        }
      }

      val csc = new CostSensitiveClassifier[String, String](5, 1, Some(scaleRange))
      csc.train(dataset, costMatrix)
      if (epoch == 0) {
        policy.learned.csc = csc
      } else {
        policy.learned.csc.avgWeights += csc.avgWeights
      }
    }
  }
}
