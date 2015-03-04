package edu.arizona.sista.discourse.rstparser.experimental

import edu.arizona.sista.discourse.rstparser._
import edu.arizona.sista.processors.Document

class ExpertPolicy(val relModel: RelationClassifier) extends Policy {
  def getNextState(currState: State,
    goldTree: DiscourseTree,
    edus: Array[Array[(Int, Int)]],
    doc: Document
  ): State = {
    val nextStatesWithMergedIndex = getNextStatesWithMergedIndex(currState, doc, edus, relModel)
    val (nextStates, mergedIndex) = nextStatesWithMergedIndex.unzip
    val costs = getStatesCosts(nextStates, goldTree)
    val (nextState, cost) = getCheapestState(nextStates, costs)
    nextState
  }
}
