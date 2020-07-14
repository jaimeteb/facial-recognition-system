import sys
import logging

if __name__ == "__main__":
    try:
        from lib.Augmentate import seq
        from lib.FaceDetector import FaceDetectorDlib
        from lib.FaceNetKeras import get_features, get_features_tolist
        from lib.FaceAlign import get_aligned_face, get_landmarks_visualization
        from lib.SSRNET_model import SSR_net, SSR_net_general
        from lib.emotion_estimator import get_emotion_estimator_model

        from lib.SVMClassifier import Classifier as SVMClassifier
        from lib.TreeClassifier import Classifier as TreeClassifier

        svm_clf = SVMClassifier()
        svm_clf.load()
        logging.info("\t\tSVMClassifier loaded")
        tree_clf = TreeClassifier()
        tree_clf.load()
        logging.info("\t\tTreeClassifier loaded")

        sys.exit(0)

    except Exception as e:
        logging.error(e)

        sys.exit(1)
