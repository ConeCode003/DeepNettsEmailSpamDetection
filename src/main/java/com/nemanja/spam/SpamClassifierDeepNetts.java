package com.nemanja.spam;

import deepnetts.data.DataSets;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.data.norm.MinMaxScaler;
import deepnetts.eval.Evaluators;
import javax.visrec.ml.eval.EvaluationMetrics;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ri.ml.classification.FeedForwardNetBinaryClassifier;




public class SpamClassifierDeepNetts {

    public static void main(String[] args)
            throws DeepNettsException, IOException {
        
        // broj feature-a je 5, target je 1 (is_spam 1 - spam, 0 - nije spam) 
        int numInputs = 5;
        int numOutputs = 1;

        // Ucitavanje csv fajla
        DataSet dataSet = DataSets.readCsv(
                "datasets/spam_detection_dataset.csv",
                numInputs,
                numOutputs,
                true,
                ","
        );
        System.out.println("Broj poruka u datasetu: " + dataSet.size());
        
        // Podela na train/test/validacioni skup 70/15/15
        DataSet<MLDataItem>[] trainRemaining = dataSet.split(0.70,0.30);
        DataSet<MLDataItem> trainSet = trainRemaining[0];
        DataSet<MLDataItem> remainingSet = trainRemaining[1];
        
       
        // Druga podela: preostalih 30% iz remaining delimo na 15% validacionih, i 15% testnih podataka
        DataSet<MLDataItem>[] validationTest = remainingSet.split(0.50,0.50);
        DataSet<MLDataItem> validationSet = validationTest[0];
        DataSet<MLDataItem> testSet = validationTest[1];
        System.out.println("Train skup: " + trainSet.size());
        System.out.println("Validation skup: "+ validationSet.size());
        System.out.println("Test skup: "+ testSet.size());
        
        /* Komentar: Nakon podele skupa proverena je zastupljenost klasa
           i utvrđeno je da je procenat spam poruka približno jednak u trening, validacionom i test skupu
        */
        
        
        // Normalizujemo/skaliramo podatke na train/test/validation skupu
        MinMaxScaler scaler = new MinMaxScaler(trainSet);
        scaler.apply(trainSet);
        scaler.apply(validationSet);
        scaler.apply(testSet);
        
        ///////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////// INICIJALNA KONFIGURACIJA MREZE 5 - 32 - 16 - 1 /////////////////////////////////////////
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(32, ActivationType.RELU)
                .addFullyConnectedLayer(16, ActivationType.RELU)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(42)
                .build();
       
            
       //Podesavanje algoritma za treniranje
       BackpropagationTrainer trainer = neuralNet.getTrainer();
       trainer.setStopEpochs(50)
               .setStopError(0.000001f)
               .setLearningRate(0.001f)
               .setOptimizer(OptimizerType.SGD)
               .setBatchMode(true)
               .setBatchSize(64)
               .setEarlyStopping(false);
       trainer.setExtendedLogging(true);
       System.out.println("");
       System.out.println("TRENIRANJE KONFIGURACIJE:  5 -> 32 -> 16 -> 1");
       //loss grafikon
       
       
       // Loss chart
       LossChart.attach(trainer, neuralNet, trainSet, validationSet);
       // Treniranje na trening skupu
       neuralNet.train(trainSet);
       System.out.println("TRENIRANJE KONFIGURACIJE:  5 -> 32 -> 16 -> 1 JE ZAVRSENO.");
       System.out.println("");
       
 
       
       ///////////////////////////////////////////////////////////////////////////////////////////////////////
       ////////////// DRUGA KONFIGURACIJA MREZE 5 - 16 - 1  //////////////////////////////////////////////////
       FeedForwardNetwork neuralNet2 = FeedForwardNetwork.builder()
        .addInputLayer(numInputs)
        .addFullyConnectedLayer(16, ActivationType.RELU)
        .addOutputLayer(numOutputs, ActivationType.SIGMOID)
        .lossFunction(LossType.CROSS_ENTROPY)
        .randomSeed(42)
        .build();
       
       BackpropagationTrainer trainer2 = neuralNet2.getTrainer();
       trainer2.setStopEpochs(50)
        .setStopError(0.000001f)
        .setLearningRate(0.001f)
        .setOptimizer(OptimizerType.SGD)
        .setBatchMode(true)
        .setBatchSize(64)
        .setEarlyStopping(false);
       
       System.out.println();
       System.out.println("TRENIRANJE KONFIGURACIJE: 5 -> 16 -> 1");
       neuralNet2.train(trainSet);
       System.out.println( "TRENIRANJE KONFIGURACIJE: 5 -> 16 -> 1 JE ZAVRSENO.");
       System.out.println();
       
       ///////////////////////////////////////////////////////////////////////////////////////////////////////
       ////////////// TRECA KONFIGURACIJA MREZE 5 - 64 - 32 - 1  /////////////////////////////////////////////
       ///////////////////////////////////////////////////////////////////////////////////////////////////////
       FeedForwardNetwork neuralNet3 = FeedForwardNetwork.builder()
        .addInputLayer(numInputs)
        .addFullyConnectedLayer(64, ActivationType.RELU)
        .addFullyConnectedLayer(32, ActivationType.RELU)
        .addOutputLayer(numOutputs, ActivationType.SIGMOID)
        .lossFunction(LossType.CROSS_ENTROPY)
        .randomSeed(42)
        .build();
       
       BackpropagationTrainer trainer3 = neuralNet3.getTrainer();
       trainer3.setStopEpochs(50)
        .setStopError(0.000001f)
        .setLearningRate(0.001f)
        .setOptimizer(OptimizerType.SGD)
        .setBatchMode(true)
        .setBatchSize(64)
        .setEarlyStopping(false);
       
       System.out.println();
       System.out.println("TRENIRANJE KONFIGURACIJE: 5 -> 64 -> 32 -> 1");
       neuralNet3.train(trainSet);
       System.out.println( "TRENIRANJE KONFIGURACIJE: 5 -> 64 -> 32 -> 1 JE ZAVRSENO.");
       System.out.println();
       
       
       ///////////////////////////////////////////////////////////////////////////////////////////////////////
       /////////////////////REZULTATI KONFIGURACIJA///////////////////////////////////////////////////////////
       ///////////////////////////////////////////////////////////////////////////////////////////////////////
       // Evaluacija na validacionom skupu
       System.out.println("REZULTATI KONFIGURACIJE:  5 -> 32 -> 16 -> 1 NA VALIDACIONOM SKUPU:");
       EvaluationMetrics validationMetrics = Evaluators.evaluateClassifier(neuralNet, validationSet);
       System.out.println(validationMetrics);
       System.out.println();
       System.out.println("REZULTATI KONFIGURACIJE: 5 -> 16 -> 1 NA VALIDACIONOM SKUPU:");
       EvaluationMetrics validationMetrics2 =
       Evaluators.evaluateClassifier(neuralNet2,validationSet);
       System.out.println(validationMetrics2);
       System.out.println("REZULTATI KONFIGURACIJE: 5 -> 64 -> 32 -> 1 NA VALIDACIONOM SKUPU:");
       EvaluationMetrics validationMetrics3 =
       Evaluators.evaluateClassifier(neuralNet3,validationSet);
       System.out.println(validationMetrics3);
       
       System.out.println();
       System.out.println("IZABRANA KONFIGURACIJA:5 -> 32 -> 16 -> 1");
       
       // Kreiranje binarnog klasifikatora od istrenirane mreze
       BinaryClassifier<float[]> binClassifier = new FeedForwardNetBinaryClassifier(neuralNet);
       
       // Prvi primer iz test skupa
       System.out.println("PREDIKCIJA ZA PRVU PORUKU IZ TEST SKUPA:");
       float[] testEmail = testSet.get(0).getInput().getValues();
       
       // Verovatnoca da je poruka spam
       Float result = binClassifier.classify(testEmail);
       System.out.println("Spam probability: " + result);
       System.out.println();
       
       //Evaluacija na test skupu
       System.out.println("REZULTATI KONFIGURACIJE:  5 -> 32 -> 16 -> 1 NA TEST SKUPU:");
       EvaluationMetrics testMetrics = Evaluators.evaluateClassifier(neuralNet,testSet);
       System.out.println(testMetrics);
       

       
    }
}
