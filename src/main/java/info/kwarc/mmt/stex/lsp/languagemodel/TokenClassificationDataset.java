package info.kwarc.mmt.stex.lsp.languagemodel;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.util.Progress;

import java.util.List;

public class TokenClassificationDataset extends RandomAccessDataset {

    private final HuggingFaceTokenizer tokenizer;
    private final List<String[]> tokens;
    private final List<int[]> labels;

    private TokenClassificationDataset(TokenClassificationDataset.Builder builder) {
        super(builder);
        this.tokenizer = builder.tokenizer;
        this.tokens = builder.tokens;
        this.labels = builder.labels;
    }

    public static Builder builder() {
        return new TokenClassificationDataset.Builder();
    }

    @Override
    public long size() {
        return tokens.size();
    }

    @Override
    public Record get(NDManager manager, long index) {
        Encoding encoding = tokenizer.encode(tokens.get((int) index));
        NDArray inputIds = manager.create(encoding.getIds()).toType(DataType.INT64, false);
        long[] wordIds = encoding.getWordIds();
        int[] alignedLabels = new int[wordIds.length];
        int[] currentLabels = labels.get((int) index);
        for (int i = 0; i < wordIds.length; i++) {
            alignedLabels[i] = wordIds[i] < 0 ? 0 : currentLabels[(int) wordIds[i]];
        }
        NDArray labelsArray = manager.create(alignedLabels);
        // return new Record(new NDList(inputIds, typeIds, attentionMask), new NDList(labelsArray.reshape(-1, 1)));
        // return new Record(new NDList(inputIds), new NDList(labelsArray.reshape(-1, 1)));
        return new Record(new NDList(inputIds), new NDList(labelsArray.reshape(-1, 1)));
    }

    @Override
    protected long availableSize() {
        return 0;
    }

    @Override
    public void prepare(Progress progress) {
    }

    public static final class Builder extends BaseBuilder<TokenClassificationDataset.Builder> {

        private List<String[]> tokens;
        private List<int[]> labels;
        private HuggingFaceTokenizer tokenizer;

        @Override
        protected Builder self() {
            return this;
        }

        Builder setTokenizer(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder setData(List<String[]> tokens, List<int[]> labels) {
            this.tokens = tokens;
            this.labels = labels;
            return this;
        }

        TokenClassificationDataset build() {
            optDataBatchifier(PaddingStackBatchifier.builder()
                    .optIncludeValidLengths(false)
                    .addPad(0, 0, (m) -> m.zeros(new Shape(1)))
                    .build());
            optLabelBatchifier(PaddingStackBatchifier.builder()
                    .optIncludeValidLengths(false)
                    .addPad(0, 0, (m) -> m.zeros(new Shape(1, 1)))
                    .build());
            return new TokenClassificationDataset(this);
        }
    }
}
