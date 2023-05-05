package info.kwarc.mmt.stex.lsp.languagemodel;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.jni.CharSpan;
import ai.djl.huggingface.tokenizers.jni.TokenizersLibrary;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.ParameterList;
import ai.djl.nn.core.Embedding;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtSymbolBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.slf4j.impl.SimpleLogger;

import java.io.*;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class HuggingFaceTokenModel implements AutoCloseable {

    public boolean isPaused = false;
    public HuggingFaceTokenizer tokenizer;
    ZooModel<NDList, NDList> model;

    public static void main(String[] args) throws IOException {
        // TODO: this is experimental! mainly due to the input format
        System.out.println("Note: Invoking the model with the main method is experimental!");
        System.setProperty(SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "error");
        ArgumentParser parser = ArgumentParsers.newFor("Main")
                .build()
                .defaultHelp(true)
                .description("Run NER on sTeX documents (or any other text).");
        parser.addArgument("<model>")
                .help("Specify PyTorch model to use.");
        parser.addArgument("<input_file>")
                .help("Newline separated input blocks (output from parser), can be blank or '-' for stdin.");
        parser.addArgument("-t", "--output_threshold")
                .help("Threshold to show prediction. Default is 0.5, 0 for all predictions.")
                .type(Float.class)
                .setDefault(0.5f);
        Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        // System.out.println("version = " + Main.class.getPackage().getSpecificationVersion());
        String inputFile = ns.getString("<input_file>");
        String[] inputWords;
        try (InputStream inputStream = inputFile.equals("-") ? System.in : new FileInputStream(inputFile);
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))
        ) {
            ArrayList<String> lines = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
            inputWords = lines.toArray(String[]::new);
        }

        // todo
        try (HuggingFaceTokenModel huggingFaceTokenModel = new HuggingFaceTokenModel()) {
            huggingFaceTokenModel.initialize(ns.getString("<model>"));
            List<LinkedList<SinglePrediction>> groupedPredictions = huggingFaceTokenModel.predict(inputWords).groupedPredictions;
            float outputThreshold = ns.getFloat("output_threshold");
            for (int i = 0; i < groupedPredictions.size(); i++) {
                LinkedList<SinglePrediction> group = groupedPredictions.get(i);
                StringBuilder block = new StringBuilder();
                for (SinglePrediction sp : group) {
                    if (sp.positiveCertainty >= outputThreshold) {
                        block.append(String.format("  %.2f  %s\n", sp.positiveCertainty, sp.token));
                    }
                }
                if (!block.isEmpty()) {
                    System.out.printf("Block %d\n", i);
                    System.out.print(block);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Calculates the maximum number of sub-word tokens from a list of wordIds
     */
    private static int getMaxSubWordTokens(long[] wordIds) {
        return IntStream.range(0, wordIds.length)
                .boxed()
                .filter(i -> wordIds[i] >= 0)
                .collect(Collectors.groupingBy(
                        i -> wordIds[i],
                        Collectors.mapping(i -> i, Collectors.collectingAndThen(
                                Collectors.summarizingInt(Integer::intValue),
                                IntSummaryStatistics::getCount)
                        ))
                ).values().stream().max(Long::compareTo).get().intValue();
    }

    public static float hausdorffDistance(NDArray set1, NDArray set2) {
        float maxDistance = 0f;
        for (int i = 0; i < set1.size(0); i++) {
            for (int j = 0; j < set2.size(0); j++) {
                NDArray diff = set1.get(i).sub(set2.get(j));
                float distance = diff.norm().getFloat();
                if (distance > maxDistance) {
                    maxDistance = distance;
                }
            }
        }
        return maxDistance;
    }

    public boolean isReady() {
        return !isPaused && tokenizer != null && model != null;
    }

    public void initialize(String filePath) throws IOException, ModelNotFoundException, MalformedModelException {
        if (filePath.endsWith(".zip")) {
            // initialize model and tokenizer from zip file contents
            initializeZip(filePath);
        } else {
            // initialize model from trained params
            Path path = Path.of(filePath);
            String filename = path.getFileName().toString();
            String modelName = filename.substring(0, filename.lastIndexOf('-'));
            Criteria<NDList, NDList> criteria = Criteria.builder()
                    .setTypes(NDList.class, NDList.class)
                    .optEngine("PyTorch")
                    .optBlock(new PtSymbolBlock((PtNDManager) PtEngine.getInstance().newBaseManager()))
                    .optModelPath(path.getParent())
                    .optModelName(modelName)
                    .build();
            model = criteria.loadModel();
            // initialize default tokenizer
            tokenizer = HuggingFaceTokenizer.builder()
                    .optTokenizerName("distilbert-base-uncased")
                    .optTruncation(true)
                    .optMaxLength(512)
                    .build();
        }
    }

    private void initializeZip(String filePath) throws IOException, MalformedModelException, ModelNotFoundException {
        try (ZipFile zip = new ZipFile(filePath)) {
            ZipEntry tokenizerEntry = zip.getEntry("tokenizer.json");
            tokenizer = HuggingFaceTokenizer.newInstance(zip.getInputStream(tokenizerEntry), null);
            ZipEntry modelEntry = zip.getEntry("model_traced.pt");
            InputStream modelInputStream = zip.getInputStream(modelEntry);
            // load from zip stream
            Model m = PtEngine.getInstance().newModel("model", null);
            m.load(modelInputStream);
            model = new ZooModel<>(m, new NoopTranslator());
        }
    }

    /**
     * This method computes the distance of one word to a list of words w.r.t. BERT embeddings
     */
    public List<Map.Entry<Long, Float>> argsortEmbeddingDistance(String word, String[] compare) {
        return argsortEmbeddingDistance(word, compare, true, false, false);
    }

    public List<Map.Entry<Long, Float>> argsortEmbeddingDistance(String word, String[] compare, boolean applyPositionalEmbeddings, boolean applyLayerNorm, boolean applyDropout) {
        // combine all strings
        String[] allPreTokens = new String[compare.length + 1];
        allPreTokens[0] = word;
        System.arraycopy(compare, 0, allPreTokens, 1, compare.length);
        // get word embeddings map
        Map<Long, NDArray> wordEmbeddingMeans = getWordEmbeddings(allPreTokens, applyPositionalEmbeddings, applyLayerNorm, applyDropout);
        // split first embeddings
        NDArray firstEmbeddingsMean = wordEmbeddingMeans.remove(0L);
        // compute distances
        Map<Long, Float> distMap = wordEmbeddingMeans.entrySet().stream()
                .map(e -> Map.entry(e.getKey() - 1, firstEmbeddingsMean.sub(e.getValue()).norm().getFloat()))
                //.map(e -> Map.entry(e.getKey() - 1, hausdorffDistance(firstEmbeddingsMean, e.getValue())))
                .collect(HashMap::new, (m, e) -> m.put(e.getKey(), e.getValue()), Map::putAll);
        // sort result by distance
        return distMap.entrySet().stream().sorted(Map.Entry.comparingByValue()).collect(Collectors.toList());
    }

    /**
     * This method computes BERT embeddings for an array of strings
     * <p>
     * NOTE: other potential solution for more complex operations: trace a method after fine-tuning into the model,
     * then use
     * <code>IValueUtils.runMethod((PtSymbolBlock) model.getBlock(), "method_name", IValue.from());</code>
     *
     * @see <a href="https://discuss.pytorch.org/t/separate-torchscript-model-into-two-parts/102774">here</a>
     */
    private Map<Long, NDArray> getWordEmbeddings(String[] preTokens) {
        return getWordEmbeddings(preTokens, true, false, false);
    }

    private Map<Long, NDArray> getWordEmbeddings(String[] preTokens, boolean applyPositionalEmbeddings, boolean applyLayerNorm, boolean applyDropout) {
        String tokenizerStrategy = TokenizersLibrary.LIB.getTruncationStrategy(tokenizer.getHandle()).toUpperCase();
        int tokenizerStride = TokenizersLibrary.LIB.getStride(tokenizer.getHandle());
        int tokenizerMaxLength = TokenizersLibrary.LIB.getMaxLength(tokenizer.getHandle());
        TokenizersLibrary.LIB.disableTruncation(tokenizer.getHandle());

        Encoding encoding = tokenizer.encode(preTokens);
        TokenizersLibrary.LIB.setTruncation(tokenizer.getHandle(), tokenizerMaxLength, tokenizerStrategy, tokenizerStride);

        long[] inputIds = encoding.getIds();
        long[] wordIds = encoding.getWordIds();

        ParameterList parameters = model.getBlock().getParameters();
        NDArray embeddingWeights = parameters.get(0).getValue().getArray();
        NDArray positionalWeights = parameters.get(1).getValue().getArray();
        NDArray normWeights = parameters.get(2).getValue().getArray();
        NDArray normBias = parameters.get(3).getValue().getArray();

        NDArray inputIdsArray = model.getNDManager().create(inputIds);
        NDArray embeddings = Embedding.embedding(inputIdsArray, embeddingWeights, SparseFormat.COO).get(0);

        int maxSubWordTokens = applyPositionalEmbeddings ? getMaxSubWordTokens(wordIds) : 0;
        NDArray maxPositionalEncodings = model.getNDManager().arange(maxSubWordTokens).add(1);
        NDArray positional = applyPositionalEmbeddings ? Embedding.embedding(maxPositionalEncodings, positionalWeights, SparseFormat.COO).get(0) : null;
        // word to embedding mapping
        Shape layerNormShape = embeddings.getShape().slice(1);
        Map<Long, NDArray> wordEmbeddingMeans = IntStream.range(0, wordIds.length)
                .boxed()
                .filter(i -> wordIds[i] >= 0)
                .collect(Collectors.groupingBy(
                        i -> wordIds[i],
                        Collectors.mapping(i -> i, Collectors.collectingAndThen(
                                Collectors.summarizingInt(Integer::intValue),
                                stats -> {
                                    int dim = stats.getMax() - stats.getMin() + 1;
                                    String idcs = String.format("%d:%d", stats.getMin(), stats.getMax() + 1);
                                    String posIdcs = String.format(":%d", dim);
                                    NDArray output = embeddings.get(idcs);
                                    if (applyPositionalEmbeddings) {
                                        output = output.add(positional.get(posIdcs));
                                    }
                                    if (applyLayerNorm) {
                                        output = LayerNorm.layerNorm(output, layerNormShape, normWeights, normBias, 1e-12f).get(0);
                                    }
                                    if (applyDropout) {
                                        output = Dropout.dropout(output, 0.1f, false).get(0);
                                    }
                                    // TODO: reconsider mean
                                    return output.mean(new int[]{0});
                                })
                        )
                ));
        return wordEmbeddingMeans;
    }

    public PredictionResult predict(String[] inputBlocks) throws TranslateException {
        // TODO: still not perfect for handling both cases: single string and split by blocks.
        //  at least not for roberta, which acts differently on special chars,
        //  i.e. "(hello)" can be a single token, which is stupid...
        Encoding encoding = tokenizer.encode(inputBlocks);
        try (NDManager manager = model.getNDManager().newSubManager()) {
            NDList result = predictInternal(manager, encoding);
            float[] posPredictions = result.get(0).get(0).softmax(-1).get(":,1").toFloatArray();
            long[] wordIds = encoding.getWordIds();
            String[] tokens = encoding.getTokens();
            CharSpan[] charTokenSpans = encoding.getCharTokenSpans();
            boolean isRobertaTokenizer = Arrays.stream(tokens).anyMatch(t -> t.startsWith("▁"));

            // group tokens by input blocks aka wordId
            List<LinkedList<SinglePrediction>> groupedPredictions = new ArrayList<>();
            for (int i = 0; i < wordIds.length; i++) {
                if (wordIds[i] < 0) {
                    continue;
                }
                int wordId = (int) wordIds[i];
                if (groupedPredictions.size() <= wordId) {
                    groupedPredictions.add(new LinkedList<>());
                }
                groupedPredictions.get(wordId).add(new SinglePrediction(tokens[i], posPredictions[i], charTokenSpans[i]));
            }
            // merge sub-word tokens
            for (int i = 0; i < groupedPredictions.size(); i++) {
                LinkedList<SinglePrediction> group = groupedPredictions.get(i);
                if (group.size() == 1) {
                    SinglePrediction sp = group.getFirst();
                    if (sp.token.startsWith("▁")) {
                        sp.token = sp.token.substring(1);
                    }
                    continue;
                }
                LinkedList<SinglePrediction> afterMerge = new LinkedList<>();
                SinglePrediction walker;
                StringBuilder mergeToken = null;
                List<Float> mergeCertainties = null;
                CharSpan mergeCharSpan = null;
                while (true) {
                    walker = group.pollFirst();
                    boolean isStart = mergeCertainties == null;
                    boolean isEnd = walker == null;
                    if (isEnd) {
                        float meanPositiveCertainty = (float) mergeCertainties.stream().mapToDouble(Float::doubleValue).average().getAsDouble();
                        afterMerge.add(new SinglePrediction(mergeToken.toString(), meanPositiveCertainty, mergeCharSpan));
                        break;
                    }
                    boolean isSubword = isRobertaTokenizer ? !walker.token.startsWith("▁") : walker.token.startsWith("##");
                    if (!isStart && !isSubword) {
                        float meanPositiveCertainty = mergeCertainties.size() == 1
                                ? mergeCertainties.get(0)
                                : (float) mergeCertainties.stream().mapToDouble(Float::doubleValue).average().getAsDouble();
                        afterMerge.add(new SinglePrediction(mergeToken.toString(), meanPositiveCertainty, mergeCharSpan));
                    }
                    if (!isSubword || isStart) {
                        String token = isRobertaTokenizer ? walker.token.substring(1) : walker.token;
                        mergeToken = new StringBuilder().append(token);
                        mergeCertainties = new ArrayList<>();
                        mergeCertainties.add(walker.positiveCertainty);
                        mergeCharSpan = walker.charSpan;
                    } else {
                        String token = isRobertaTokenizer ? walker.token : walker.token.substring(2);
                        mergeToken.append(token);
                        mergeCertainties.add(walker.positiveCertainty);
                        mergeCharSpan = new CharSpan(mergeCharSpan.getStart(), walker.charSpan.getEnd());
                    }
                }
                groupedPredictions.set(i, afterMerge);
            }
            return new PredictionResult(groupedPredictions, encoding.getTokens().length);
        }
    }

    private NDList predictInternal(NDManager manager, Encoding encoding) throws TranslateException {
        // handles different model inputs, i.e., BERT, DistilBERT, SciBERT, RoBERTa
        NDList modelInput = new NDList();
        modelInput.add(manager.create(encoding.getIds()).expandDims(0));
        modelInput.add(manager.create(encoding.getAttentionMask()).expandDims(0));
        modelInput.add(manager.create(encoding.getTypeIds()).expandDims(0));
        try {
            return model.newPredictor().predict(modelInput);
        } catch (TranslateException e) {
            Matcher m = Pattern.compile("Expected at most (\\d) argument").matcher(e.getMessage());
            if (m.find()) {
                for (int i = 0; i < Integer.parseInt(m.group(1)); i++) {
                    modelInput.remove(modelInput.size() - 1);
                }
                return model.newPredictor().predict(modelInput);
            } else {
                throw e;
            }
        }
    }

    public TrainingResult finetune(String ptModelPath, List<String[]> tokens, List<int[]> labels) {
        return finetune(ptModelPath, tokens, labels, 16);
    }

    public TrainingResult finetune(String ptModelPath, List<String[]> tokens, List<int[]> labels, int batchSize) {
        // TODO: only DistilBERT tested so far
        TrainingResult trainingResult;
        Criteria<NDList, NDList> modelCriteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelUrls(ptModelPath)
                .optProgress(new ProgressBar())
                .optEngine("PyTorch")
                .optOption("trainParam", "true")
                .build();
        HuggingFaceTokenizer.Builder tokenizerBuilder = HuggingFaceTokenizer.builder()
                .optTokenizerName("distilbert-base-uncased")
                .optTruncation(true)
                .optPadding(true)
                //.optPadToMaxLength()
                .optMaxLength(512);
        try (
                Model ptModel = modelCriteria.loadModel();
                HuggingFaceTokenizer tokenizer = tokenizerBuilder.build()
        ) {
            RandomAccessDataset fullDataset = TokenClassificationDataset
                    .builder()
                    .setSampling(batchSize, true)
                    .setTokenizer(tokenizer)
                    .setData(tokens, labels)
                    .build();
            // validation does not work with DJL currently, so 7:0 instead of 7:3 --> no validation
            RandomAccessDataset[] datasets = fullDataset.randomSplit(7, 0);
            RandomAccessDataset trainingSet = datasets[0];
            RandomAccessDataset validationSet = datasets[1];

            DefaultTrainingConfig config = setupTrainingConfig();
            try (Trainer trainer = ptModel.newTrainer(config)) {
                trainer.setMetrics(new Metrics());
                // initialize trainer with proper input shape
                trainer.initialize(new Shape(batchSize, 512));

                EasyTrain.fit(trainer, 3, trainingSet, validationSet);
                trainingResult = trainer.getTrainingResult();
            }
            ptModel.save(Path.of("."), "distilbert");
        } catch (ModelNotFoundException | MalformedModelException | IOException | TranslateException e) {
            throw new RuntimeException(e);
        }
        return trainingResult;
    }

    private DefaultTrainingConfig setupTrainingConfig() {
        /*TrainingListener listener = new LoggingTrainingListener() {
            @Override
            public void onTrainingBatch(Trainer trainer, BatchData batchData) {
                super.onTrainingBatch(trainer, batchData);
                float loss = trainer.getLoss().evaluate(
                        batchData.getBatch().getLabels(),
                        batchData.getPredictions().get(batchData.getPredictions().keySet().toArray()[0])
                ).getFloat();
                Double accuracy = trainer.getMetrics().latestMetric("train_all_A").getValue();
                Double recall = trainer.getMetrics().latestMetric("train_all_R").getValue();
                Double precision = trainer.getMetrics().latestMetric("train_all_P").getValue();
                System.out.printf("\nAccuracy=%.3f, Recall=%.3f, Precision=%.3f, Loss=%.3f\n",
                        accuracy, recall, precision, loss);
            }
        };*/
        Adam optimizer = Optimizer.adam().optWeightDecays(0.01f).optLearningRateTracker(Tracker.fixed(2e-5f)).build();
        //NDArray classWeights = NDManager.newBaseManager().create(new float[]{0.1f, 0.9f});
        //return new DefaultTrainingConfig(new BalancedSoftmaxCrossEntropyLoss(classWeights))
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss("L", 1, -1, true, true))
                .addEvaluator(new Accuracy("A", -1))
                .addEvaluator(new RecallEvaluator("R", -1))
                .addEvaluator(new PrecisionEvaluator("P", -1))
                .optDevices(Engine.getInstance().getDevices())
                .addTrainingListeners(TrainingListener.Defaults.logging("./logs/"))
                //.addTrainingListeners(listener)
                .optOptimizer(optimizer);
    }

    @Override
    public void close() {
        if (model != null) {
            model.close();
        }
        if (tokenizer != null) {
            tokenizer.close();
        }
    }
}
