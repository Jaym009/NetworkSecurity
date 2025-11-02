from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
import sys
from sklearn.metrics import f1_score, precision_score, recall_score

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        f1score = f1_score(y_true, y_pred)
        prec_score = precision_score(y_true, y_pred)
        rec_score = recall_score(y_true, y_pred)
        classification_metric_artifact = ClassificationMetricArtifact(
            f1_score=f1score,
            precision_score=prec_score,
            recall_score=rec_score
        )
        return classification_metric_artifact
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e