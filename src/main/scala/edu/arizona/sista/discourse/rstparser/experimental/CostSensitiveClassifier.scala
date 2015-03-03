package edu.arizona.sista.discourse.rstparser.experimental

import scala.math.sqrt
import edu.arizona.sista.learning._
import edu.arizona.sista.struct.{ Counter, Lexicon }
import edu.arizona.sista.discourse.rstparser.StructureClassifier.{ LOWER, UPPER }
import breeze.linalg._

class CostSensitiveClassifier[L, F](
    val epochs: Int,
    val aggressiveness: Double,
    val scaleRange: Option[ScaleRange[F]] = None
) {
  require(epochs > 0, "'epochs' should be greater than zero")
  require(aggressiveness > 0, "'aggressiveness' should be greater than zero")

  // import static methods
  import CostSensitiveClassifier._

  // uninitialized lexicons
  var labelLexicon: Lexicon[L] = _
  var featureLexicon: Lexicon[F] = _
  var avgWeights: DenseMatrix[Double] = _

  // uninitialized variables
  var numLabels: Int = _
  var numFeatures: Int = _
  var numSamples: Int = _

  // train with default costs
  def train(dataset: Dataset[L, F]): Unit =
    train(dataset, mkCostMatrix(dataset))

  def train(dataset: Dataset[L, F], costMatrix: DenseMatrix[Double]): Unit = {
    numLabels = dataset.numLabels
    numFeatures = dataset.numFeatures
    numSamples = dataset.size

    val indices = DenseVector.range(0, numSamples)
    val weights = DenseMatrix.zeros[Double](numLabels, numFeatures)
    avgWeights = weights.copy

    // initialize lexicons
    labelLexicon = Lexicon(dataset.labelLexicon)
    featureLexicon = Lexicon(dataset.featureLexicon)

    for (epoch <- 0 until epochs; i <- shuffle(indices).values) {
      val datum = dataset.mkDatum(i)
      val trueLabel = labelLexicon.get(datum.label).get
      val feats = mkFeatureVector(datum.featuresCounter)
      val predLabel = argmax(weights * feats)
      val predCost = costMatrix(i, predLabel)

      if (predCost > 0) {
        val loss = weights(predLabel, ::) * feats - weights(trueLabel, ::) * feats + sqrt(predCost)
        val learningRate = loss / ((feats dot feats) + (1 / (2 * aggressiveness)))
        val update = feats.t :* learningRate
        weights(trueLabel, ::) :+= update
        weights(predLabel, ::) :-= update
      }

      avgWeights += weights
    }
  }

  def predictScores(datum: Datum[L, F]): DenseVector[Double] =
    predictScores(datum.featuresCounter)

  def predictScores(counter: Counter[F]): DenseVector[Double] =
    avgWeights * mkFeatureVector(counter)

  def mkFeatureVector(counter: Counter[F]): SparseVector[Double] = {
    val scaledFeats = if (scaleRange.isDefined) {
      Datasets.svmScaleDatum(counter, scaleRange.get, LOWER, UPPER)
    } else {
      counter
    }
    val indexData = for {
      (f, v) <- scaledFeats.toSeq
      i <- featureLexicon.get(f)
    } yield (i, v)
    val (index, data) = indexData.sortBy(_._1).unzip
    new SparseVector(index.toArray, data.toArray, numFeatures)
  }

  def predictLabeledScores(counter: Counter[F]): Seq[(L, Double)] =
    predictScores(counter).valuesIterator.zipWithIndex.map {
      case (s, i) => (labelLexicon.get(i), s)
    }.toSeq

  def predictLabel(datum: Datum[L, F]): L =
    predictLabel(datum.featuresCounter)

  def predictLabel(counter: Counter[F]): L =
    labelLexicon.get(argmax(predictScores(counter)))
}

object CostSensitiveClassifier {
  // returns a cost matrix for the given dataset
  // with cost 0 for the right label and cost 1 for everything else
  def mkCostMatrix[L, F](dataset: Dataset[L, F]): DenseMatrix[Double] =
    DenseMatrix.tabulate[Double](dataset.size, dataset.numLabels) {
      (i, j) => if (dataset.labels(i) == j) 0 else 1
    }
}
