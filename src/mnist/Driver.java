package mnist;

public class Driver {
	
	static final String TRAIN_IMAGES = "train-images.idx3-ubyte";
	static final String TRAIN_LABELS = "train-labels.idx1-ubyte";
	static final int SIZE = 60000;

	public static void main(String[] args) {
		
		ImageReader trainReader = new ImageReader(TRAIN_IMAGES, TRAIN_LABELS, 10000);
		
		FFNN network = new FFNN(trainReader.getImages());
		network.process();
	}
	
	
}
