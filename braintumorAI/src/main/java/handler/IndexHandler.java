package handler;

import java.io.IOException;

import javax.faces.bean.ManagedBean;
import javax.faces.bean.SessionScoped;
import javax.faces.context.FacesContext;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.primefaces.model.UploadedFile;

import lombok.Getter;
import lombok.Setter;

@ManagedBean(name = "indexHandler")
@SessionScoped
public class IndexHandler {
	@Getter
	@Setter
	private UploadedFile uploadedFile;
	@Getter
	@Setter
	private double result;
	
	public void handleFileUpload() {
		if(uploadedFile != null)
			compute();
        
    }
	
	private void compute() {
		try {
			MultiLayerNetwork model;
			try {
				model = KerasModelImport.importKerasSequentialModelAndWeights(this.getClass().getResourceAsStream("model.hdf5"));
				NativeImageLoader loader = new NativeImageLoader(150, 150, 3);
				INDArray input = loader.asMatrix(uploadedFile.getInputstream());
		        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
		        preProcessor.transform(input);
		        result = model.output(input).getDouble(0,0);
		        System.out.println(model.predict(input));
		        FacesContext.getCurrentInstance().getExternalContext().redirect("result.xhtml");
			} catch (InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
				e.printStackTrace();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	public boolean isTumor() {
		return result > 0.6;
	}
}
