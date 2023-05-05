package info.kwarc.mmt.stex.lsp.languagemodel;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.util.Pair;

import java.util.concurrent.ConcurrentHashMap;

public class RecallEvaluator extends Evaluator {
    private final ConcurrentHashMap<String, Long> truePositives;
    private final ConcurrentHashMap<String, Long> falseNegatives;
    private final int axis;

    public RecallEvaluator(String name) {
        this(name, -1);
    }

    public RecallEvaluator(String name, int axis) {
        super(name);
        truePositives = new ConcurrentHashMap<>();
        falseNegatives = new ConcurrentHashMap<>();
        this.axis = axis;
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        return null;
    }

    @Override
    public void addAccumulator(String key) {
        totalInstances.put(key, 0L);
        truePositives.put(key, 0L);
        falseNegatives.put(key, 0L);
    }

    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        Pair<Long, Pair<Long, Long>> update = helper(labels, predictions);
        totalInstances.compute(key, (k, v) -> v + update.getKey());
        truePositives.compute(key, (k, v) -> v + update.getValue().getKey());
        falseNegatives.compute(key, (k, v) -> v + update.getValue().getValue());
    }

    @Override
    public void resetAccumulator(String key) {
        totalInstances.compute(key, (k, v) -> 0L);
        truePositives.compute(key, (k, v) -> 0L);
        falseNegatives.compute(key, (k, v) -> 0L);
    }

    @Override
    public float getAccumulator(String key) {
        return (float) truePositives.get(key) / (truePositives.get(key) + falseNegatives.get(key));
    }

    protected Pair<Long, Pair<Long, Long>> helper(NDList labels, NDList predictions) {
        NDArray label = labels.head();
        NDArray prediction = predictions.head();
        checkLabelShapes(label, prediction);
        NDArray predictionReduced;
        if (!label.getShape().equals(prediction.getShape())) {
            // Multi-class, sparse label
            predictionReduced = prediction.argMax(axis);
            predictionReduced = predictionReduced.reshape(label.getShape());
        } else {
            // Multi-class, one-hot label
            predictionReduced = prediction;
        }
        long total = label.size();
        try (NDArray nd = label.toType(DataType.INT64, true)) {
            Long truePositives = predictionReduced.toType(DataType.INT64, false).logicalAnd(nd).countNonzero().getLong();
            Long falseNegatives = predictionReduced.toType(DataType.INT64, false).logicalNot().logicalAnd(nd).countNonzero().getLong();
            return new Pair<>(total, new Pair<>(truePositives, falseNegatives));
        }
    }
}

