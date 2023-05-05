package info.kwarc.mmt.stex.lsp.languagemodel;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.util.Pair;

import java.util.concurrent.ConcurrentHashMap;

class PrecisionEvaluator extends Evaluator {
    private final ConcurrentHashMap<String, Long> truePositives;
    private final ConcurrentHashMap<String, Long> falsePositives;
    private final int axis;

    public PrecisionEvaluator(String name) {
        this(name, -1);
    }

    public PrecisionEvaluator(String name, int axis) {
        super(name);
        truePositives = new ConcurrentHashMap<>();
        falsePositives = new ConcurrentHashMap<>();
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
        falsePositives.put(key, 0L);
    }

    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        Pair<Long, Pair<Long, Long>> update = helper(labels, predictions);
        totalInstances.compute(key, (k, v) -> v + update.getKey());
        truePositives.compute(key, (k, v) -> v + update.getValue().getKey());
        falsePositives.compute(key, (k, v) -> v + update.getValue().getValue());
    }

    @Override
    public void resetAccumulator(String key) {
        totalInstances.compute(key, (k, v) -> 0L);
        truePositives.compute(key, (k, v) -> 0L);
        falsePositives.compute(key, (k, v) -> 0L);
    }

    @Override
    public float getAccumulator(String key) {
        return (float) truePositives.get(key) / (truePositives.get(key) + falsePositives.get(key));
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
            Long falsePositives = predictionReduced.toType(DataType.INT64, false).logicalAnd(nd.logicalNot()).countNonzero().getLong();
            return new Pair<>(total, new Pair<>(truePositives, falsePositives));
        }
    }
}
