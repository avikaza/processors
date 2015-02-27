package edu.arizona.sista.discourse.rstparser.experimental

import edu.arizona.sista.discourse.rstparser._
import edu.arizona.sista.processors.Document
import edu.arizona.sista.discourse.rstparser.Utils.mkGoldEDUs
import breeze.linalg.SparseVector

class LearnedPolicy(val weights: SparseVector[Double], val corpusStats: CorpusStats, val relModel: RelationClassifier) extends Policy {
  val featureExtractor = new FeatureExtractor

  def getNextState(currState: State, goldTree: DiscourseTree, doc: Document): State = {
    val edus = mkGoldEDUs(goldTree, doc)
    val scores = 0 to currState.size - 2 map { i =>
      val features = getFeatures(currState, i, doc, edus, corpusStats)
      weights dot features
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
                  corpusStats: CorpusStats): SparseVector[Double] = {
    val left = state(merge)
    val right = state(merge + 1)
    val d = relModel.mkDatum(left, right, doc, edus, StructureClassifier.NEG)
    val ld = relModel.classOf(d)
    val (label, dir) = relModel.parseLabel(ld)
    featureExtractor.getFeatures(left, right, doc, edus, corpusStats, label)
  }

  def parseWithGoldEDUs(tree: DiscourseTree, doc: Document): DiscourseTree =
    getCompletePath(tree, doc).last.head
}