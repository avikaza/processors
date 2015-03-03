package edu.arizona.sista.discourse.rstparser.experimental

import edu.arizona.sista.discourse.rstparser._
import edu.arizona.sista.learning.Datasets

object Trainer extends App {
  val trainDirName = "/Users/marcov/data/RST_cached_preprocessing/rst_train"
  val testDirName = "/Users/marcov/data/RST_cached_preprocessing/rst_test"
  val dependencySyntax = true
  val processor = CacheReader.getProcessor(dependencySyntax)

  val rstparser = RSTParser.loadFrom(RSTParser.DEFAULT_DEPENDENCYSYNTAX_MODEL_PATH)

  var policy: LearnedPolicy = _

  println("training ...")
  train()
  println("testing ...")
  test()

  def train(): Unit = {
    val (treedocs, corpusStats) = RSTParser.mkTrees(trainDirName, processor)
    // val structureClassifier = new StructureClassifier
    // val structDataset = structureClassifier.mkDataset(treedocs)
    // val scaleRanges = Datasets.svmScaleDataset(
      // structDataset,
      // lower = StructureClassifier.LOWER,
      // upper = StructureClassifier.UPPER
    // )
    val scaleRanges = rstparser.structModel.scaleRanges
    val searn = new Searn(3, 0.1, rstparser.relModel)
    searn.train(treedocs.toIndexedSeq, corpusStats, scaleRanges)
    policy = searn.policy.learned
  }

  def test(): Unit = {
    val scorer = new DiscourseScorer
    val structScoreGold = new DiscourseScore()
    val (treedocs, corpusStats) = RSTParser.mkTrees(testDirName, processor)
    for ((tree, doc) <- treedocs) {
      val sys = policy.parseWithGoldEDUs(tree, doc)
      scorer.score(sys, tree, structScoreGold, ScoreType.OnlyStructure)
    }
    println("STRUCT SCORE (with gold EDUs):\n" + structScoreGold)
  }
}
