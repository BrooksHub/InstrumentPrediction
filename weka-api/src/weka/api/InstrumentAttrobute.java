package weka.api;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.*;


public class InstrumentAttrobute {

	public static void main(String[] args) throws Exception {

		DataSource sourceTrain = new DataSource("/Users/CyberKabuki/Downloads/InstrumentData/trainData/train.csv");
		DataSource sourceTest = new DataSource("/Users/CyberKabuki/Downloads/InstrumentData/testData/test.csv");
		ArffSaver saver = new ArffSaver();
		Instances train = sourceTrain.getDataSet();
		Instances test = sourceTest.getDataSet();

		// Is instrument being played ?
		// - loop through array of instruments
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

		int count=0;

		for (String instrument:instruments){
			System.out.println("workimg on instrument"+" "+ instrument); 

			// Add  

			Add filter;
			filter = new Add();
			filter.setAttributeIndex("last");
			filter.setNominalLabels("TRUE,FALSE");
			filter.setAttributeName("has"+ instrument);
			filter.setInputFormat(train);
			Instances newDataTrain = Filter.useFilter(train, filter);

			filter = new Add();
			filter.setAttributeIndex("last");
			filter.setNominalLabels("TRUE,FALSE");
			filter.setAttributeName("has"+ instrument);
			filter.setInputFormat(test);
			Instances newDataTest = Filter.useFilter(test, filter);


			// select instrument
			String instrumentSelected = instrument;

			// Set TRUE or FALSE values
			for (int i = 0; i < newDataTrain.numInstances(); i++) {
				// index of labels TRUE:0,FALSE:1

				String wholeRow = newDataTrain.instance(i).toString();

				if(wholeRow.toUpperCase().contains(instruments.get(count).toUpperCase())) {

					newDataTrain.instance(i).setValue(newDataTrain.numAttributes() - 1, "TRUE");

				} else {
					newDataTrain.instance(i).setValue(newDataTrain.numAttributes() - 1, "FALSE");

				}

			} 
			/// remove unwanted attributes train

			//remove 1st attribute
			//set options to remove attribute
			String[] opts = new String[]{"-R",String.valueOf(newDataTrain.numAttributes()-1)};
			System.out.println(opts.toString());
			//create remove filter
			Remove r = new Remove();
			//set filter options
			r.setOptions(opts);
			//pass dataset to filter
			r.setInputFormat(newDataTrain);
			//apply filter
			Instances removeDataTrain1 = Filter.useFilter(newDataTrain, r);

			//remove 2nd attribute
			//set options to remove attribute
			String[] opts2 = new String[]{"-R",String.valueOf(removeDataTrain1.numAttributes()-1)};
			System.out.println(opts2.toString());
			//create remove filter
			Remove r2 = new Remove();
			//set filter options
			r2.setOptions(opts2);
			//pass dataset to filter
			r2.setInputFormat(removeDataTrain1);
			//apply filter
			Instances removeDataTrain2 = Filter.useFilter(removeDataTrain1, r2);

			///remove unwanted attributes test

			//remove 1st attribute
			//set options to remove attribute
			String[] optsTest1 = new String[]{"-R",String.valueOf(newDataTest.numAttributes()-1)};
			System.out.println(optsTest1.toString());
			//create remove filter
			Remove rt1 = new Remove();
			//set filter options
			rt1.setOptions(optsTest1);
			//pass dataset to filter
			rt1.setInputFormat(newDataTest);
			//apply filter
			Instances removeDataTest1 = Filter.useFilter(newDataTest, rt1);

			//remove 2nd attribute
			//set options to remove attribute
			String[] optsTest2 = new String[]{"-R",String.valueOf(removeDataTest1.numAttributes()-1)};
			System.out.println(optsTest2.toString());
			//create remove filter
			Remove rt2 = new Remove();
			//set filter options
			rt2.setOptions(optsTest2);
			//pass dataset to filter
			rt2.setInputFormat(removeDataTest1);
			//apply filter
			Instances removeDataTest2 = Filter.useFilter(removeDataTest1, rt2);

			//remove 3nrd attribute
			//set options to remove attribute
			String[] optsTest3 = new String[]{"-R",String.valueOf(removeDataTest2.numAttributes()-1)};
			System.out.println(optsTest3.toString());
			//create remove filter
			Remove rt3 = new Remove();
			//set filter options
			rt3.setOptions(optsTest3);
			//pass dataset to filter
			rt3.setInputFormat(removeDataTest2);
			//apply filter
			Instances removeDataTest3 = Filter.useFilter(removeDataTest2, rt3);

			//remove 4th attribute
			//set options to remove attribute
			String[] optsTest4 = new String[]{"-R",String.valueOf(removeDataTest3.numAttributes()-1)};
			System.out.println(optsTest4.toString());
			//create remove filter
			Remove rt4 = new Remove();
			//set filter options
			rt4.setOptions(optsTest4);
			//pass dataset to filter
			rt4.setInputFormat(removeDataTest3);
			//apply filter
			Instances removeDataTest4 = Filter.useFilter(removeDataTest3, rt4);

			//remove 5th attribute
			//set options to remove attribute
			String[] optsTest5 = new String[]{"-R",String.valueOf(removeDataTest4.numAttributes()-1)};
			System.out.println(optsTest5.toString());
			//create remove filter
			Remove rt5 = new Remove();
			//set filter options
			rt5.setOptions(optsTest5);
			//pass dataset to filter
			rt5.setInputFormat(removeDataTest4);
			//apply filter
			Instances removeDataTest5 = Filter.useFilter(removeDataTest4, rt5);

			// save new data for each instrument 
			saver.setInstances(removeDataTrain2);
			saver.setFile(new File("/Users/CyberKabuki/Downloads/InstrumentData/trainInstruments/"+instrumentSelected+".arff"));
			saver.writeBatch();

			// save new data for each test
			saver.setInstances(removeDataTest5);
			saver.setFile(new File("/Users/CyberKabuki/Downloads/InstrumentData/testInstruments/"+instrumentSelected+"Test.arff"));
			saver.writeBatch();

			count++;
		}
	}
}
