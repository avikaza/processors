package edu.arizona.sista.discourse.rstparser.experimental

import scala.util.Random
import edu.arizona.sista.discourse.rstparser._
import edu.arizona.sista.processors.Document

class InterpolatedPolicy(val relModel: RelationClassifier) extends Policy {
  var expertProbability: Double = 1
  val expert: ExpertPolicy = new ExpertPolicy(relModel)
  var learned: LearnedPolicy = _

  def getNextState(currState: State, goldTree: DiscourseTree, edus: Array[Array[(Int, Int)]], doc: Document): State =
    policy.getNextState(currState, goldTree, edus, doc)

  def policy: Policy =
    if (Random.nextDouble() <= expertProbability) expert else learned
}
