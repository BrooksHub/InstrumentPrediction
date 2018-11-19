package weka.api;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SimpleLogistic;

public class Classifier {


	public static void main(String[] args) {

		List<String> instruments = new ArrayList<>();

		// Add new elements to the ArrayList
		instruments.add("Accordian");
		instruments.add("Clarinet");
		instruments.add("DoubleBass");
		instruments.add("Oboe");
		instruments.add("Piano");
		instruments.add("Saxophone");
		instruments.add("Violin");
		instruments.add("Cello");
		instruments.add("Tuba");
		instruments.add("Viola");
		instruments.add("Trombone");
		instruments.add("Trumpet");
		
		try {

			for (String instrument:instruments){
				System.out.println("workimg on instrument"+" "+ instrument); 
				
				String selectInstrument = instrument;
				
				// loading data from an ARFF file
				DataSource trainSource = new DataSource("/Users/CyberKabuki/Downloads/InstrumentData/trainInstruments/"+selectInstrument+".arff");
				Instances trainData = trainSource.getDataSet();

				DataSource testSource = new DataSource("/Users/CyberKabuki/Downloads/InstrumentData/testInstruments/"+selectInstrument+"Test.arff");
				Instances testData = testSource.getDataSet();

				// setting class attribute if the data format does not provide this information
				// For example, the XRFF format saves the class attribute information as well
				if (trainData.classIndex() == -1)
					trainData.setClassIndex(trainData.numAttributes() - 1);

				if (testData.classIndex() == -1)
					testData.setClassIndex(testData.numAttributes() - 1);

				// defining a classifier

				SimpleLogistic sl = new SimpleLogistic(); 
				
				// evaluation via cross-validation
				Evaluation eval = new Evaluation(trainData);
				eval.crossValidateModel(sl, trainData, 10, new Random(1));
				System.out.println(eval.toSummaryString("\nResults\n======\n", false));

				// training the classifier and testing it on test data
				sl.buildClassifier(trainData);
				for (int i = 0; i < testData.numInstances(); i++) {
					double clsLabel = sl.classifyInstance(testData.instance(i));
					testData.instance(i).setClassValue(clsLabel);
					
					// print your test dataset with predicted labels at the end
					System.out.println(clsLabel);
				}

			
				// save new data for each test
				ArffSaver saver = new ArffSaver();
				saver.setInstances(testData);
				saver.setFile(new File("/Users/CyberKabuki/Downloads/InstrumentData/Classifier/SimpleLogistic/"+selectInstrument+"SL.arff"));
				saver.writeBatch();
				
			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}


