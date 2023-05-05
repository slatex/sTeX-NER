package info.kwarc.mmt.stex.lsp.languagemodel;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;

public class BalancedSoftmaxCrossEntropyLoss extends SoftmaxCrossEntropyLoss {

    private final int classAxis;
    private final boolean sparseLabel;
    private final boolean fromLogit;
    private final NDArray weights;

    public BalancedSoftmaxCrossEntropyLoss(NDArray weights) {
        this("Loss", weights, -1, true, true);
    }

    public BalancedSoftmaxCrossEntropyLoss(String name, NDArray weights, int classAxis, boolean sparseLabel, boolean fromLogit) {
        super(name);
        this.weights = weights;
        this.classAxis = classAxis;
        this.sparseLabel = sparseLabel;
        this.fromLogit = fromLogit;
    }

    @Override
    public NDArray evaluate(NDList label, NDList prediction) {
        NDArray pred = prediction.singletonOrThrow();
        if (fromLogit) {
            pred = pred.logSoftmax(classAxis);
        }
        NDArray loss;
        NDArray lab = label.singletonOrThrow();
        if (sparseLabel) {
            // loss = pred.mul(weights).neg().sum(new int[]{classAxis}, true).div(weights.sum());
            Shape predShape = pred.getShape();
            int numClasses = (int) predShape.get((classAxis + predShape.dimension()) % predShape.dimension());
            NDArray labOneHot = lab.oneHot(numClasses).reshape(pred.getShape());
            loss = pred.mul(labOneHot).mul(weights).neg().sum().div(predShape.get(1) * 2);
        } else {
            lab = lab.reshape(pred.getShape());
            loss = pred.mul(lab).neg().sum(new int[]{classAxis}, true);
        }
        /*if (weights != null) {
            loss = loss.mul(weight);
        }*/
        return loss.mean();
    }
}
