import cv2
import edgeiq
import urllib.request

def main():
    obj_det = edgeiq.ObjectDetection("tester2204/CE-Recog")
    if edgeiq.is_jetson():
    	obj_det.load(engine=edgeiq.Engine.DNN_CUDA)
    	print("Nvidia Jetson Detected\n")
    else:
    	obj_det.load(engine=edgeiq.Engine.DNN)
    	print("Device is not a Nvidia Jetson Board\n")
    print("Initializing Application...\n")
    print("Model:\n{}\n".format(obj_det.model_id))
    print("Engine:\n{}\n".format(obj_det.engine))
    print("Labels:\n{}\n".format(obj_det.labels))
    
    #imgURL = "https://specials-images.forbesimg.com/imageserve/5e88b867e2bb040006427704/0x0.jpg"
    #urllib.request.urlretrieve(imgURL, "this.jpg") #Change based on OS and User
    
    #image = "Images/this.jpg"
    
    image_lists = sorted(list(edgeiq.list_images("Images/")))
    
    with edgeiq.Streamer(queue_depth=len(image_lists), inter_msg_time=7) as streamer:
    	i = 0
    	while i < 3:
    		for image_list in image_lists:
    			show_image = cv2.imread(image_list)
    			image = show_image.copy()
    		
    			results = obj_det.detect_objects(image, confidence_level=.5)
    		
    			image = edgeiq.markup_image(image, results.predictions, colors=obj_det.colors)
    		
    			shown = ["Model: {}".format(obj_det.model_id)]
    			shown.append("Inference time: {:1.3f} s".format(results.duration))
    			shown.append("Objects:")
    		
    			for prediction in results.predictions:
    				shown.append("{}: {:2.2f}%".format(prediction.label, prediction.confidence * 100))
    			streamer.send_data(image, shown)
    		streamer.wait()
    		i = i+1
    		
    		
    
    #if streamer.check_exit():
    print("That's it folks!")
    print("Thanks for using Ben's Object Recognition Model & Software")
    print("Sponsored by: Darien's Face")

if __name__ == "__main__":
    main()
