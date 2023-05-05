package info.kwarc.mmt.stex.lsp.languagemodel;

import ai.djl.huggingface.tokenizers.jni.CharSpan;

public class SinglePrediction {
    public String token;
    public float positiveCertainty;
    public CharSpan charSpan;

    public SinglePrediction(String token, float positiveCertainty, CharSpan charSpan) {
        this.token = token;
        this.positiveCertainty = positiveCertainty;
        this.charSpan = charSpan;
    }
}
