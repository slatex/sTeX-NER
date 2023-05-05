package info.kwarc.mmt.stex.lsp.languagemodel;

import java.util.LinkedList;
import java.util.List;

public class PredictionResult {
    public List<LinkedList<SinglePrediction>> groupedPredictions;
    public int numTokens;

    public PredictionResult(List<LinkedList<SinglePrediction>> groupedPredictions, int numTokens) {
        this.groupedPredictions = groupedPredictions;
        this.numTokens = numTokens;
    }
}
