import os
from selfcal_omr.detector import OMRDetector
from selfcal_omr.helper import Make_Normalized_Sheet, draw_contours



def test_isolation_detect_runs():
    detector = OMRDetector()
    test_image = os.path.join("tests", "inputs", "223.jpg")
    detector = OMRDetector()
    
    #Detection with Isolation Forest algorithm without any preassumption of bubble size
    img, blocks, missing, contours, points, grid_dx, grid_dy, teta, rate = detector.isolation_detect(test_image)
    
    print("grid_dx: ", grid_dx)
    print("grid_dy: ", grid_dy)
    print("teta: ", teta , "success_rate: ", rate)
    Make_Normalized_Sheet(blocks, points , missing, img)
    draw_contours(img, contours, missing)

