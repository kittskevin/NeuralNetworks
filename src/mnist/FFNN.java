package mnist;

import java.util.List;
import java.util.Random;

public class FFNN {
	
	static final int INPUT = 784;
	static final int HIDDEN = 15;
	static final int OUTPUT = 10;
	static final double factor = 1.0;
	static final double THRESHHOLD = .8;
	
	private double[][] inWeights;
	private double[][] outWeights;
	
	List<Image> images;
	
	public FFNN(List<Image> images) {
		this.images = images;	
		inWeights = new double[INPUT][HIDDEN];
		outWeights = new double[HIDDEN][OUTPUT];
		
		// Initialize weights randomly
		Random r = new Random();
		for (int i = 0; i < inWeights.length; i++) {
			for (int j = 0; j < inWeights[i].length; j++) {
				inWeights[i][j] = r.nextDouble() / 2;
			}
		}
		for (int i = 0; i < outWeights.length; i++) {
			for (int j = 0; j < outWeights[i].length; j++) {
				outWeights[i][j] = r.nextDouble() / 2;
			}
		}
		
	}

	public void process() {
		int index = 0;
		int correct = 0;
		while (index < 9999) {
			Image image = images.get(index);
			double[] out = propigate(image);
			
			System.out.println();
			for (int i = 0; i < out.length; i++) {
				System.out.printf("%f ", out[i]);
			}
			System.out.println();
			System.out.printf("Guessed: %d%n", findMax(out));
			System.out.printf("Expected: %d%n", image.label);
			
			if (findMax(out) == image.label) {
				correct++;
			}
			
			index++;
		}
		System.out.println(correct);
	}
	
	private double[] propigate(Image image) {
		
		// Get the first neuron outputs
		double[] inVals = new double[INPUT];  // INPUTS == pixels.length
		for (int i = 0; i < INPUT; i++) {
			inVals[i] = (image.pixels[i] + 1) / 255.0;
		}
		
		// Add the hidden sums and run through function
		double[] hiddenVals = new double[HIDDEN];
		double[] hiddenSums = new double[HIDDEN];
		for (int i = 0; i < HIDDEN; i++) {
			double hiddenSum = 0;
			for (int j = 0; j < INPUT; j++) {  // Cache locality?
				hiddenSum += inVals[j] * inWeights[j][i];
			}
			hiddenSums[i] = hiddenSum;
			hiddenVals[i] = sigmoid(hiddenSum);
		}
		
		// Get the output sums and run through function
		double[] outputVals = new double[OUTPUT];
		double[] outputSums = new double[OUTPUT];
		for (int i = 0; i < OUTPUT; i++) {
			double outputSum = 0;
			for (int j = 0; j < HIDDEN; j++) {
				if (hiddenVals[j] > THRESHHOLD) {
					outputSum += hiddenVals[j] * outWeights[j][i];  // Cache locality?
				}
				
			}
			outputSums[i] = outputSum;
			outputVals[i] = sigmoid(outputSum);
		}
		
		
		
		// Back propigate
		double[] error = new double[OUTPUT];
		double[] expected = new double[OUTPUT];
		expected[image.label] = 1;
			
		for (int i = 0; i < OUTPUT; i++) {
			error[i] = expected[i] - outputVals[i];
		}
		
		double[] deltaOutputSum = new double[OUTPUT];
		for (int j = 0; j < OUTPUT; j++) {
			deltaOutputSum[j] = (sigmoidPrime(outputSums[j]) * error[j]);
		}

		
		double[] deltaHiddenSumTotal = new double[HIDDEN];
		double[][] deltaHiddenSum = new double[OUTPUT][HIDDEN];
		
		for (int i = 0; i < OUTPUT; i++) {
			for (int j = 0; j < HIDDEN; j++) {
				deltaHiddenSum[i][j] = (deltaOutputSum[i] / outWeights[j][i]) * sigmoidPrime(hiddenSums[j]);
				deltaHiddenSumTotal[j] += deltaHiddenSum[i][j];
			}
		}
		
		
		// Update weights
		for (int i = 0; i < HIDDEN; i++) {
			for (int j = 0; j < OUTPUT; j++) {
				outWeights[i][j] += (deltaOutputSum[j] / hiddenVals[i]) * factor * .1;
			}
		}
//		for (int i = 0; i < INPUT; i++) {
//			for (int j = 0; j < HIDDEN; j++) {
//				if (inVals[i] > 1) {
//					inWeights[i][j] += ((deltaHiddenSumTotal[j] / (OUTPUT * 1.0)) / inVals[i]) * factor;
//				}		
//			}
//		}
		
		return outputVals;
		
	}
	
	private static double sigmoid(double x) {
		return (1.0 / (1.0 + Math.exp(-x)));
	}
	
	private static double sigmoidPrime(double x) {
		return Math.exp(x) / Math.pow(Math.exp(x) + 1, 2);
	}
	
	private static int findMax(double[] xs) {
		double max = xs[0];
		int index = 0;
		for (int i = 1; i < xs.length; i++) {
			if (Double.compare(max, xs[i]) < 0) {
				max = xs[i];
				index = i;
			}
		}
		return index;
	}
	
}
