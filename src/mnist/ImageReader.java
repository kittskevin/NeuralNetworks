package mnist;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ImageReader {

	private MnistManager manager;
	private List<Image> images;
	private int count;

	public ImageReader(String imagesFile, String labelsFile, int count) {
		
		this.count = count;
		try {
			manager = new MnistManager(imagesFile, labelsFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		long start = System.currentTimeMillis();
		images = new ArrayList<Image>();
		int index = 1;
		while (index < this.count) {
			try {
				int[] pixels = manager.readImage();
				int label = manager.readLabel();
				
				images.add(new Image(pixels, label));

				index++;
				manager.setCurrent(index);
				
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		long finish = System.currentTimeMillis();
		System.out.printf("Images Read: %d%n", images.size());
		System.out.printf("Took: %d ms%n", finish - start);
	}
	
	public List<Image> getImages() {
		return images;
	}
		
	
}
