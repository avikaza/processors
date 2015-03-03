package edu.arizona.sista.discourse.rstparser.experimental

import edu.arizona.sista.discourse.rstparser._
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.processors.Document
import edu.arizona.sista.discourse.rstparser.Utils.mkGoldEDUs
import breeze.linalg._

class LearnedPolicy(
    var csc: CostSensitiveClassifier[String, String],
    val corpusStats: CorpusStats,
    val relModel: RelationClassifier
) extends Policy {
  val featureExtractor = new RelationFeatureExtractor

  def getNextState(currState: State, goldTree: DiscourseTree, doc: Document): State = {
    val edus = mkGoldEDUs(goldTree, doc)
    val scores = 0 to currState.size - 2 map { i =>
      val features = getFeatures(currState, i, doc, edus, corpusStats)
      val labeledScores = csc.predictLabeledScores(features)
      labeledScores.find(_._1 == StructureClassifier.POS) match {
        case Some((label, score)) => score
        case None => sys.error("something went wrong")
      }
    }
    val merge = scores indexOf scores.max
    val children = currState.slice(merge, merge + 2).toArray
    val d = relModel.mkDatum(children(0), children(1), doc, edus, StructureClassifier.NEG)
    val ld = relModel.classOf(d)
    val (label, dir) = relModel.parseLabel(ld)
    val node = new DiscourseTree(label, dir, children)
    val nextState = currState.take(merge) ++ Seq(node) ++ currState.drop(merge + 2)
    nextState
  }

  def getFeatures(state: State,
                  merge: Int,
                  doc: Document,
                  edus: Array[Array[(Int, Int)]],
                  corpusStats: CorpusStats): Counter[String] = {
    val left = state(merge)
    val right = state(merge + 1)
    val d = relModel.mkDatum(left, right, doc, edus, StructureClassifier.NEG)
    val ld = relModel.classOf(d)
    val (label, dir) = relModel.parseLabel(ld)
    featureExtractor.mkFeatures(left, right, doc, edus, corpusStats, label)
  }

  def parseWithGoldEDUs(tree: DiscourseTree, doc: Document): DiscourseTree =
    getCompletePath(tree, doc).last.head
}
