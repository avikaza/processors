package edu.arizona.sista.discourse.rstparser.experimental

import edu.arizona.sista.discourse.rstparser._
import edu.arizona.sista.processors.Document
import edu.arizona.sista.discourse.rstparser.Utils._

trait Policy {
  // a policy must be able to "predict" the next state
  // given the current state and the gold tree
  // NOTE the gold tree may be ignored depending on the Policy type
  def getNextState(
    currState: State,
    goldTree: DiscourseTree,
    edus: Array[Array[(Int, Int)]],
    doc: Document
  ): State

  // builds a path from the starting state to a valid result
  def getCompletePath(tree: DiscourseTree, edus: Array[Array[(Int, Int)]], doc: Document): Seq[State] = {
    @annotation.tailrec
    def mkPath(path: Seq[State]): Seq[State] =
      if (path.last.size > 1) mkPath(path :+ getNextState(path.last, tree, edus, doc))
      else path

    val startState = edusToState(edus, doc)
    mkPath(Vector(startState))
  }

  private def edusToState(edus: Array[Array[(Int, Int)]], doc: Document): State = for {
    i <- 0 until edus.size
    j <- 0 until edus(i).size
    edu = edus(i)(j)
  } yield new DiscourseTree(i, edu._1, edu._2, doc, j)
}
