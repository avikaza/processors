package edu.arizona.sista.processors.corenlp

import edu.arizona.sista.discourse.rstparser.RSTParser
import edu.arizona.sista.processors._
import edu.arizona.sista.struct._
import edu.stanford.nlp.parser.lexparser.{ParserAnnotations, LexicalizedParser}
import edu.stanford.nlp.pipeline.{ParserAnnotatorUtils, StanfordCoreNLP, Annotation}
import java.util.Properties
import collection.mutable.{ListBuffer, ArrayBuffer}
import edu.stanford.nlp.ling.CoreAnnotations._
import scala.collection.JavaConversions._
import scala.reflect.ClassTag
import edu.stanford.nlp.util.CoreMap
import edu.stanford.nlp.ling.{CoreAnnotations, CoreLabel}
import java.util
import collection.mutable
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation
import edu.stanford.nlp.trees.{GrammaticalStructureFactory, SemanticHeadFinder}
import edu.stanford.nlp.trees.{Tree => StanfordTree}
import edu.stanford.nlp.semgraph.{SemanticGraph, SemanticGraphCoreAnnotations}

/**
 * API for Stanford's CoreNLP tools
 * User: mihais
 * Date: 3/1/13
 * Last Modified: Remove laziness.
 */
class CoreNLPProcessor(val internStrings:Boolean = true,
                       val basicDependencies:Boolean = false,
                       val withDiscourse:Boolean = false,
                       val maxSentenceLength:Int = 100) extends Processor {
  val tokenizerWithoutSentenceSplitting = mkTokenizerWithoutSentenceSplitting
  val tokenizerWithSentenceSplitting = mkTokenizerWithSentenceSplitting
  val posTagger = mkPosTagger
  val lemmatizer = mkLemmatizer
  val ner = mkNer
  val coref = mkCoref
  val rstConstituentParser = CoreNLPProcessor.fetchParser(RSTParser.DEFAULT_CONSTITUENTSYNTAX_MODEL_PATH)

  //
  // we maintain our own copy of a LexicalizedParser to control which sentences are parsed
  // the CoreNLP option parser.maxlen does not work well
  //
  val stanfordParser = mkLexicalizedParser
  val gsf = mkGSF
  val headFinder = new SemanticHeadFinder()

  def mkLexicalizedParser: LexicalizedParser = {
    val parser = LexicalizedParser.loadModel()
    parser
  }

  def mkGSF:GrammaticalStructureFactory = {
    val tlp = stanfordParser.getTLPParams.treebankLanguagePack
    val g = tlp.grammaticalStructureFactory(tlp.punctuationWordRejectFilter(), tlp.typedDependencyHeadFinder())
    g
  }

  def mkTokenizerWithoutSentenceSplitting: StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize")
    new StanfordCoreNLP(props)
  }

  def mkTokenizerWithSentenceSplitting: StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit")
    new StanfordCoreNLP(props)
  }

  def mkPosTagger: StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "pos")
    new StanfordCoreNLP(props, false)
  }

  def mkLemmatizer: StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "lemma")
    new StanfordCoreNLP(props, false)
  }

  def mkNer: StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "ner")
    new StanfordCoreNLP(props, false)
  }

  def mkParser: StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "parse")
    new StanfordCoreNLP(props, false)
  }

  def mkCoref: StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "dcoref")
    new StanfordCoreNLP(props, false)
  }

  /**
   * Hook to allow postprocessing of CoreNLP tokenization
   * This is useful for domain-specific corrections, such as the ones in BioNLPProcessor
   * If you change the tokens, make sure to store them back in the sentence!
   * @param sentence Input CoreNLP sentence
   * @return The modified tokens
   */
  def postprocessTokens(sentence:CoreMap): java.util.List[CoreLabel] = {
    sentence.get(classOf[TokensAnnotation])

    //
    // Add postprocessing code here then:
    // sentence.set(classOf[TokensAnnotation], modifiedTokens)
    //
  }

  /**
   * Hook to allow the preprocessing of input text to CoreNLP
   * This is useful for domain-specific corrections, such as the ones in BioNLPProcessor, where we remove Table and Fig references
   * @param origText The original input text
   * @return The preprocessed text
   */
  def preprocessText(origText:String):String = {
    origText
  }

  def preprocessSentences(origSentences:Iterable[String]):Iterable[String] = {
    val sents = new ListBuffer[String]()
    for(os <- origSentences)
      sents += preprocessText(os)
    sents.toList
  }

  def mkDocument(origText:String): Document = {
    val text = preprocessText(origText)

    val annotation = new Annotation(text)
    tokenizerWithSentenceSplitting.annotate(annotation)
    val sas = annotation.get(classOf[SentencesAnnotation])
    val sentences = new Array[Sentence](sas.size())
    var offset = 0
    for (sa <- sas) {
      sentences(offset) = mkSentence(sa)
      offset += 1
    }

    // just in case the postprocessing code changed token offsets, reset them
    var tokenOffset = 0
    for(sa <- sas) {
      val crtTokens = sa.get(classOf[TokensAnnotation])
      sa.set(classOf[TokenBeginAnnotation], new Integer(tokenOffset))
      tokenOffset += crtTokens.size()
      sa.set(classOf[TokenEndAnnotation], new Integer(tokenOffset))
    }

    new CoreNLPDocument(sentences, Some(annotation))
  }

  def mkSentence(annotation:CoreMap): Sentence = {
    val tas = postprocessTokens(annotation)

    val wordBuffer = new ArrayBuffer[String]
    val startOffsetBuffer = new ArrayBuffer[Int]
    val endOffsetBuffer = new ArrayBuffer[Int]

    for (ta <- tas) {
      wordBuffer.add(in(ta.word))
      startOffsetBuffer += ta.beginPosition()
      endOffsetBuffer += ta.endPosition()
    }

    new Sentence(
      wordBuffer.toArray,
      startOffsetBuffer.toArray,
      endOffsetBuffer.toArray)
  }

  def in(s:String):String = {
    if (internStrings) Processor.internString(s)
    else s
  }

  def arrayOrNone[T: ClassTag](b:ArrayBuffer[T]): Option[Array[T]] = {
    if (b.size > 0) new Some[Array[T]](b.toArray)
    else None
  }

  def mkDocumentFromSentences(origSentences:Iterable[String],
                              charactersBetweenSentences:Int = 1): Document = {
    val sentences = preprocessSentences(origSentences)
    val docAnnotation = new Annotation(sentences.mkString(mkSep(charactersBetweenSentences)))
    val sentencesAnnotation = new util.ArrayList[CoreMap]()
    docAnnotation.set(classOf[SentencesAnnotation], sentencesAnnotation.asInstanceOf[java.util.List[CoreMap]])
    val docSents = new Array[Sentence](sentences.size)

    var sentOffset = 0
    var charOffset = 0
    var tokenOffset = 0
    for(sentence <- sentences) {
      val tmpAnnotation = new Annotation(sentence)
      tokenizerWithoutSentenceSplitting.annotate(tmpAnnotation)
      val crtTokens:java.util.List[CoreLabel] = postprocessTokens(tmpAnnotation)

      // construct a proper sentence, with token and character offsets adjusted to make the entire document consistent
      val crtSent = new Annotation(sentence)
      crtSent.set(classOf[TokensAnnotation], crtTokens)
      crtSent.set(classOf[TokenBeginAnnotation], new Integer(tokenOffset))
      tokenOffset += crtTokens.size()
      crtSent.set(classOf[TokenEndAnnotation], new Integer(tokenOffset))
      var i = 0
      while(i < crtTokens.size()) {
        val crtTok = crtTokens.get(i)
        crtTok.setBeginPosition(crtTok.beginPosition() + charOffset)
        crtTok.setEndPosition(crtTok.endPosition() + charOffset)
        i += 1
      }

      sentencesAnnotation.add(crtSent)
      docSents(sentOffset) = mkSentence(crtSent)

      charOffset += sentence.length + charactersBetweenSentences
      sentOffset += 1
    }

    new CoreNLPDocument(docSents, Some(docAnnotation))
  }

  private def mkSep(size:Int):String = {
    val os = new mutable.StringBuilder
    for (i <- 0 until size) os.append(" ")
    os.toString()
  }

  def mkDocumentFromTokens(sentences:Iterable[Iterable[String]],
                           charactersBetweenSentences:Int = 1,
                           charactersBetweenTokens:Int = 1): Document = {
    val sb = new ListBuffer[String]
    for (s <- sentences)
      sb += s.mkString(mkSep(charactersBetweenTokens))
    val sentenceTexts = sb.toArray
    val docAnnotation = new Annotation(sentenceTexts.mkString(mkSep(charactersBetweenSentences)))
    val sentencesAnnotation = new util.ArrayList[CoreMap]()
    docAnnotation.set(classOf[SentencesAnnotation], sentencesAnnotation.asInstanceOf[java.util.List[CoreMap]])
    val docSents = new Array[Sentence](sentences.size)

    var sentOffset = 0
    var charOffset = 0
    var tokenOffset = 0
    for(sentence <- sentences) {
      val crtTokens:util.List[CoreLabel] = new util.ArrayList[CoreLabel]()
      var tokOffset = 0
      for (w <- sentence) {
        val crtTok = new CoreLabel()
        crtTok.setWord(w)
        crtTok.setBeginPosition(charOffset)
        charOffset += w.length
        crtTok.setEndPosition(charOffset)
        crtTokens.add(crtTok)
        tokOffset += 1
        charOffset += charactersBetweenTokens
      }

      val crtSent = new Annotation(sentenceTexts(sentOffset))
      crtSent.set(classOf[TokensAnnotation], crtTokens)
      crtSent.set(classOf[TokenBeginAnnotation], new Integer(tokenOffset))
      tokenOffset += crtTokens.size()
      crtSent.set(classOf[TokenEndAnnotation], new Integer(tokenOffset))

      sentencesAnnotation.add(crtSent)
      docSents(sentOffset) = mkSentence(crtSent)
      sentOffset += 1
    }

    new CoreNLPDocument(docSents, Some(docAnnotation))
  }

  def basicSanityCheck(doc:Document, checkAnnotation:Boolean = true): Option[Annotation] = {
    if (doc.sentences == null)
      throw new RuntimeException("ERROR: Document.sentences == null!")
    if (doc.sentences.size == 0) return None
    if (doc.sentences(0).words == null)
      throw new RuntimeException("ERROR: Sentence.words == null!")

    if(checkAnnotation) {
      val annotation = doc.asInstanceOf[CoreNLPDocument].annotation.getOrElse(
        throw new RuntimeException("ERROR: annotator called after Document.clear()!"))
      Some(annotation)
    } else {
      None
    }
  }

  def postprocessTags(annotation:Annotation) {

  }

  def tagPartsOfSpeech(doc:Document) {
    val annotation = basicSanityCheck(doc)
    if (annotation.isEmpty) return

    posTagger.annotate(annotation.get)

    postprocessTags(annotation.get)

    // convert CoreNLP Annotations to our data structures
    val sas = annotation.get.get(classOf[SentencesAnnotation])
    var offset = 0
    for (sa <- sas) {
      val tb = new ArrayBuffer[String]
      val tas = sa.get(classOf[TokensAnnotation])
      for (ta <- tas) {
        tb += in(ta.tag())
      }
      doc.sentences(offset).tags = Some(tb.toArray)
      offset += 1
    }
  }

  def lemmatize(doc:Document) {
    val annotation = basicSanityCheck(doc)
    if (annotation.isEmpty) return
    if (doc.sentences.head.tags == None)
      throw new RuntimeException("ERROR: you have to run the POS tagger before lemmatization!")

    lemmatizer.annotate(annotation.get)

    val sas = annotation.get.get(classOf[SentencesAnnotation])
    var offset = 0
    for (sa <- sas) {
      val tb = new ArrayBuffer[String]
      val tas = sa.get(classOf[TokensAnnotation])
      for (ta <- tas) {
        tb += in(ta.lemma())
      }
      doc.sentences(offset).lemmas = Some(tb.toArray)
      offset += 1
    }
  }

  def namedEntitySanityCheck(doc:Document):Option[Annotation] = {
    val annotation = basicSanityCheck(doc)
    if (annotation.isEmpty) return None
    if (doc.sentences.head.tags == None)
      throw new RuntimeException("ERROR: you have to run the POS tagger before NER!")
    if (doc.sentences.head.lemmas == None)
      throw new RuntimeException("ERROR: you have to run the lemmatizer before NER!")
    annotation
  }

  def recognizeNamedEntities(doc:Document) {
    val annotation = namedEntitySanityCheck(doc)
    if(annotation.isEmpty) return

    try {
      ner.annotate(annotation.get)
    } catch {
      case e:Exception => {
        println("Caught NER exception!")
        println("Document:\n" + doc)
        throw e
      }
    }

    // convert CoreNLP Annotations to our data structures
    val sas = annotation.get.get(classOf[SentencesAnnotation])
    var offset = 0
    for (sa <- sas) {
      val tb = new ArrayBuffer[String]
      val nb = new ArrayBuffer[String]
      val tas = sa.get(classOf[TokensAnnotation])
      for (ta <- tas) {
        tb += in(ta.ner())
        val n = ta.get(classOf[NormalizedNamedEntityTagAnnotation])
        if (n != null) nb += in(n)
        else nb += in("O")
      }
      doc.sentences(offset).entities = Some(tb.toArray)
      doc.sentences(offset).norms = Some(nb.toArray)
      offset += 1
    }
  }

  def parse(doc:Document) {
    val annotation = basicSanityCheck(doc)
    if (annotation.isEmpty) return

    val sas = annotation.get.get(classOf[SentencesAnnotation])
    var offset = 0
    for (sa <- sas) {
      // run the actual parser here
      val stanfordTree = stanfordParse(sa)

      // store Stanford annotations; Stanford dependencies are created here!
      ParserAnnotatorUtils.fillInParseAnnotations(false, true, gsf, sa, stanfordTree)

      // save our own structures
      if (stanfordTree != null) {
        // save the constituent tree, including head word information
        val position = new MutableNumber[Int](0)
        doc.sentences(offset).syntacticTree = Some(toTree(stanfordTree, position))

        // save syntactic dependencies
        doc.sentences(offset).dependencies = Some(toDirectedGraph(sa))
      } else {
        doc.sentences(offset).syntacticTree = None
        doc.sentences(offset).dependencies = None
      }
      offset += 1
    }
  }

  def setWord(w:CoreLabel, label:String): Unit = {
    w.setWord(label)
    w.setLemma(label)
    w.setValue(label)
  }

  def parensToSymbols(words:java.util.List[CoreLabel]):java.util.List[CoreLabel] = {
    val processedWords = new util.ArrayList[CoreLabel]()
    for(w <- words) {
      val nw = new CoreLabel(w)
      if(nw.word() == "(") {
        setWord(nw, "-LRB-")
      }
      else if(nw.word() == ")") {
        setWord(nw, "-RRB-")
      }
      processedWords.add(nw)
    }
    processedWords
  }

  def stanfordParse(sentence:CoreMap):StanfordTree = {
    val constraints = sentence.get(classOf[ParserAnnotations.ConstraintAnnotation])
    val words = parensToSymbols(sentence.get(classOf[CoreAnnotations.TokensAnnotation]))

    var tree:StanfordTree = null

    //
    // Do not parse sentences longer than this
    // Those are likely coming from tables, so: (a) we don't know how to parse them anyway; (b) they would take forever
    //
    if(words.size < maxSentenceLength) {
      // the actual parsing
      val pq = stanfordParser.parserQuery()
      pq.setConstraints(constraints)

      //print("Parsing sentence:")
      //for(w <- words) print(s" ${w.toString()}")
      //println()

      pq.parse(words)

      // fetch the best tree
      try {
        tree = pq.getBestParse
        if (tree != null)
          tree.setScore(pq.getPCFGScore % -10000.0)
      } catch {
        case e: Exception =>
          System.err.println("WARNING: Parsing of sentence failed, possibly because of out of memory. " +
            "Will ignore and continue: " + edu.stanford.nlp.ling.Sentence.listToString(words))
      }

      //println("SYNTACTIC TREE: " + tree)
    } else {
      System.err.println("Skipping sentence of length " + words.size)
    }

    // create a fake tree if the actual parsing failed
    if(tree == null)
      tree = ParserAnnotatorUtils.xTree(words)

    //println("TREE: " + tree)
    tree
  }

  def toDirectedGraph(sa:CoreMap):DirectedGraph[String] = {
    val edgeBuffer = new ListBuffer[(Int, Int, String)]
    var da:SemanticGraph = null
    if(basicDependencies)
      da = sa.get(classOf[SemanticGraphCoreAnnotations.BasicDependenciesAnnotation])
    else
      da = sa.get(classOf[SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation])
    val edges = da.getEdgeSet
    for (edge <- edges) {
      val head:Int = edge.getGovernor.get(classOf[IndexAnnotation])
      val modifier:Int = edge.getDependent.get(classOf[IndexAnnotation])
      var label = edge.getRelation.getShortName
      val spec = edge.getRelation.getSpecific
      if (spec != null) label = label + "_" + spec
      edgeBuffer.add((head - 1, modifier - 1, in(label)))
    }

    val roots = new mutable.HashSet[Int]
    for (iw <- da.getRoots) {
      roots.add(iw.get(classOf[IndexAnnotation]) - 1)
    }

    val dg = new DirectedGraph[String](edgeBuffer.toList, roots.toSet)
    //println(dg)
    dg
  }

  def toTree(
    stanfordTree:StanfordTree,
    position:MutableNumber[Int]):Tree = {
    assert(stanfordTree != null)

    if (stanfordTree.isLeaf) {
      val tree = Terminal(stanfordTree.label.value())
      tree.setIndex(position.value)
      position.value += 1
      return tree
    }

    // println("Converting tree: " + stanfordTree.toString)
    val children = new Array[Tree](stanfordTree.numChildren())
    for (i <- 0 until stanfordTree.numChildren()) {
      children(i) = toTree(stanfordTree.getChild(i), position)
    }
    val value = stanfordTree.label.value()
    val start = children(0).startOffset
    val end = children(children.length - 1).endOffset

    val headDaughter = headFinder.determineHead(stanfordTree)
    var head = -1
    var i = 0
    while(i < stanfordTree.numChildren() && head == -1) {
      if (headDaughter == stanfordTree.getChild(i)) {
        head = i
      }
      i += 1
    }

    val nt = NonTerminal(value, children)
    nt.setStartEndIndices(start, end)
    nt.setHead(head)
    nt
  }

  def chunking(doc:Document) {
    // CoreNLP does not have shallow parsing yet
  }

  def labelSemanticRoles(doc:Document) {
    // CoreNLP does not have SRL yet
  }

  def resolveCoreference(doc:Document) {
    val annotation = basicSanityCheck(doc)
    if (annotation.isEmpty) return

    if (doc.sentences.head.tags == None)
      throw new RuntimeException("ERROR: you have to run the POS tagger before coreference resolution!")
    if (doc.sentences.head.lemmas == None)
      throw new RuntimeException("ERROR: you have to run the lemmatizer before coreference resolution!")
    if (doc.sentences.head.entities == None)
      throw new RuntimeException("ERROR: you have to run the NER before coreference resolution!")
    if(doc.sentences.head.dependencies == None)
      throw new RuntimeException("ERROR: you have to run the parser before coreference resolution!")

    coref.annotate(annotation.get)

    val chains = annotation.get.get(classOf[CorefChainAnnotation])
    val mentions = new ListBuffer[CorefMention]

    for (cid <- chains.keySet()) {
      // println("cluster " + cid)
      val mentionMap = chains.get(cid).getMentionMap
      for (mid <- mentionMap.keySet()) {
        for (mention <- mentionMap.get(mid)) {
          // val isRep = mention == cluster.getRepresentativeMention
          // println("\tmention " + mid.getSource + " " + mid.getTarget + " " + mention.startIndex + " " + mention.endIndex + " " + isRep + " [" + mention.mentionSpan + "]")

          // Processor indexes things from 0 not 1!
          val m = new CorefMention(
            mid.getSource - 1,
            mid.getTarget - 1,
            mention.startIndex - 1,
            mention.endIndex - 1,
            cid)
          mentions += m
        }
      }
    }

    doc.coreferenceChains = Some(new CorefChains(mentions.toList))
  }

  def discourse(doc:Document) {
    if(! withDiscourse) return
    basicSanityCheck(doc, checkAnnotation = false)

    if (doc.sentences.head.tags == None)
      throw new RuntimeException("ERROR: you have to run the POS tagger before discourse parsing!")
    if (doc.sentences.head.lemmas == None)
      throw new RuntimeException("ERROR: you have to run the lemmatizer before discourse parsing!")
    if(doc.sentences.head.dependencies == None)
      throw new RuntimeException("ERROR: you have to run the dependency parser before discourse parsing!")
    if(doc.sentences.head.syntacticTree == None)
      throw new RuntimeException("ERROR: you have to run the constituent parser before discourse parsing!")

    val out = rstConstituentParser.parse(doc)
    doc.discourseTree = Some(out._1)

    //println("FOUND DISCOURSE TREE:\n" + out._1)
  }
}

object CoreNLPProcessor {
  var rstParser:RSTParser = null

  def fetchParser(path:String):RSTParser = {
    this.synchronized {
      if(rstParser == null) rstParser = RSTParser.loadFrom(path)
      rstParser
    }
  }
}
