package com.nemanja.spam;

import deepnetts.data.MLDataItem;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.TrainingEvent;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.SwingUtilities;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ri.ml.classification.FeedForwardNetBinaryClassifier;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

public final class LossChart {

    private LossChart() {
    }

    public static void attach(
            BackpropagationTrainer trainer,
            FeedForwardNetwork neuralNet,
            DataSet<MLDataItem> trainSet,
            DataSet<MLDataItem> validationSet) {

        // Vrednosti koje će biti prikazane na grafikonu
        List<Integer> epochs = new ArrayList<>();
        List<Double> trainingLoss = new ArrayList<>();
        List<Double> validationLoss = new ArrayList<>();

        // Klasifikator koristi istu neuronsku mrežu i njene trenutne težine
        BinaryClassifier<float[]> classifier =
                new FeedForwardNetBinaryClassifier(neuralNet);

        // Listener se aktivira tokom procesa treniranja
        trainer.addListener(event -> {

            // Nakon završetka svake epohe računamo loss
            if (event.getType()
                    == TrainingEvent.Type.EPOCH_FINISHED) {

                int epoch = epochs.size() + 1;

                double trainError =
                        calculateBinaryCrossEntropy(
                                classifier,
                                trainSet
                        );

                double validationError =
                        calculateBinaryCrossEntropy(
                                classifier,
                                validationSet
                        );

                epochs.add(epoch);
                trainingLoss.add(trainError);
                validationLoss.add(validationError);

                System.out.printf(
                        "Epoha %d | Train loss: %.6f"
                        + " | Validation loss: %.6f%n",
                        epoch,
                        trainError,
                        validationError
                );
            }

            // Kada se trening završi, pravimo i čuvamo grafikon
            if (event.getType()
                    == TrainingEvent.Type.STOPPED) {

                createAndSaveChart(
                        epochs,
                        trainingLoss,
                        validationLoss
                );
            }
        });
    }

    
    private static double calculateBinaryCrossEntropy(
            BinaryClassifier<float[]> classifier,
            DataSet<MLDataItem> dataSet) {

        double totalLoss = 0.0;

        
        double epsilon = 0.0000001;

        for (int i = 0; i < dataSet.size(); i++) {

            MLDataItem item = dataSet.get(i);

            float[] input =
                    item.getInput().getValues();

            double actual =
                    item.getTargetOutput().getValues()[0];

            double probability =
                    classifier.classify(input);

            // Ograničavanje verovatnoće na bezbedan opseg
            probability = Math.max(
                    epsilon,
                    Math.min(1.0 - epsilon, probability)
            );

            double itemLoss = -(
                    actual * Math.log(probability)
                    + (1.0 - actual)
                    * Math.log(1.0 - probability)
            );

            totalLoss += itemLoss;
        }

        return totalLoss / dataSet.size();
    }

    
    private static void createAndSaveChart(
            List<Integer> epochs,
            List<Double> trainingLoss,
            List<Double> validationLoss) {

        if (epochs.isEmpty()) {
            System.err.println(
                    "Loss grafikon nije napravljen jer nema zabeleženih epoha."
            );
            return;
        }

        XYChart chart = new XYChartBuilder()
                .width(900)
                .height(600)
                .title("Loss kroz epohe - mreza 5-32-16-1")
                .xAxisTitle("Epoha")
                .yAxisTitle("Binary Cross-Entropy Loss")
                .build();

        chart.addSeries(
                "Train loss",
                epochs,
                trainingLoss
        );

        chart.addSeries(
                "Validation loss",
                epochs,
                validationLoss
        );

        try {
            BitmapEncoder.saveBitmap(
                    chart,
                    "loss_curve",
                    BitmapFormat.PNG
            );

            System.out.println();
            System.out.println(
                    "Loss grafikon je sacuvan kao loss_curve.png"
            );

        } catch (IOException exception) {
            System.err.println(
                    "Grafikon nije sacuvan: "
                    + exception.getMessage()
            );
        }

        // Prikaz grafikona u posebnom prozoru
        SwingUtilities.invokeLater(() ->
                new SwingWrapper<>(chart).displayChart()
        );
    }
}