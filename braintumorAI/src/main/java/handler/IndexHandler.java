package handler;

import java.io.IOException;

import javax.faces.application.FacesMessage;
import javax.faces.bean.ManagedBean;
import javax.faces.bean.SessionScoped;
import javax.faces.context.FacesContext;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.primefaces.event.FileUploadEvent;
import org.primefaces.model.UploadedFile;

import lombok.Getter;
import lombok.Setter;

@ManagedBean(name = "indexHandler")
@SessionScoped
public class IndexHandler {
	@Getter
	@Setter
	private UploadedFile uploadedFile;
	
	public void handleFileUpload(FileUploadEvent event) {
        FacesMessage msg = new FacesMessage("Successful", event.getFile().getFileName() + " is uploaded.");
        FacesContext.getCurrentInstance().addMessage(null, msg);
        compute();
        
    }
	
	private void compute() {
		try {
			MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(this.getClass().getResourceAsStream("model.hdf5"));
			NativeImageLoader loader = new NativeImageLoader(150, 150, 3);
			INDArray input = loader.asMatrix(uploadedFile);
	        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
	        preProcessor.transform(input);
			model.output(input, false);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
}
