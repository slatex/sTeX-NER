# sTeX-NER

This project is part of a Master's Thesis with the objective to identify missing semantic annotations in sTeX
documents using machine learning.
More specifically, BERT-based transformer language models are fine-tuned on flexiformal sTeX content in a token
classification task.

A single drop-in artifact in form of a Java Archive can be generated and used with the
[sTeX-IDE](https://github.com/slatex/sTeX-IDE).


## Development

If you open this repository with Intellij Idea you will have simple gradle integration.

Otherwise, simply build a JAR artifact using

```sh
./gradlew jar
```

The result will be written to `build/libs/`.

