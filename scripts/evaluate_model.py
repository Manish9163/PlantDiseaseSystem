import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
import json

class ModelEvaluator:
    def __init__(self, model_path='best_plant_disease_model.h5', dataset_path="plantvillage dataset/color"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.class_names = []
        self.label_encoder = LabelEncoder()
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_test_data(self, test_size=0.2):
        """Load and preprocess test data"""
        print("Loading test data...")
        
        images = []
        labels = []
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.dataset_path) 
                     if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        self.class_names = sorted(class_dirs)
        print(f"Found {len(self.class_names)} classes")
        
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {class_name}: {len(image_files)} images")
            
            for img_file in image_files:
                try:
                    img_path = os.path.join(class_path, img_file)
                    img = self.preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(class_name)
                except Exception as e:
                    continue
        
        X = np.array(images)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(self.class_names))
        
        print(f"Test dataset loaded: {X.shape[0]} images")
        return X, y_categorical, y
    
    def preprocess_image(self, img_path):
        """Preprocess image for evaluation"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            # Advanced preprocessing
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            img = img.astype(np.float32) / 255.0
            return img
        except Exception as e:
            return None
    
    def evaluate_model(self, X_test, y_test, y_true_labels):
        """Comprehensive model evaluation"""
        print("Evaluating model...")
        
        # Make predictions
        if len(self.model.inputs) == 3:  # Ensemble model
            y_pred = self.model.predict([X_test, X_test, X_test])
        else:  # Single model
            y_pred = self.model.predict(X_test)
        
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average='weighted'
        )
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        report = classification_report(
            y_true_classes, y_pred_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Print per-class metrics
        print("\nPer-Class Performance:")
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true_classes,
            'y_pred': y_pred_classes,
            'y_pred_proba': y_pred,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot detailed confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(25, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Plant Disease Recognition', fontsize=16)
        plt.xlabel('Predicted Disease', fontsize=12)
        plt.ylabel('Actual Disease', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate and display per-class accuracy
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nPer-Class Accuracy (Diagonal of Normalized Confusion Matrix):")
        for i, class_name in enumerate(self.class_names):
            accuracy = cm_normalized[i, i]
            print(f"{class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    def plot_class_performance(self, report, save_path='class_performance.png'):
        """Plot per-class performance metrics"""
        classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1 = [report[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Disease Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_top_predictions(self, y_pred_proba, y_true, num_samples=10):
        """Plot top predictions for sample images"""
        # Get random samples
        indices = np.random.choice(len(y_pred_proba), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            true_class = self.class_names[y_true[idx]]
            pred_proba = y_pred_proba[idx]
            
            # Get top 5 predictions
            top_5_indices = np.argsort(pred_proba)[-5:][::-1]
            top_5_classes = [self.class_names[j] for j in top_5_indices]
            top_5_probs = [pred_proba[j] for j in top_5_indices]
            
            # Plot
            axes[i].barh(range(5), top_5_probs, alpha=0.7)
            axes[i].set_yticks(range(5))
            axes[i].set_yticklabels(top_5_classes)
            axes[i].set_xlabel('Probability')
            axes[i].set_title(f'True: {true_class}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('top_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, results, save_path='evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        report = {
            'model_info': {
                'model_path': self.model_path,
                'num_classes': len(self.class_names),
                'classes': self.class_names
            },
            'overall_metrics': {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score'])
            },
            'per_class_metrics': {}
        }
        
        # Add per-class metrics
        for class_name in self.class_names:
            if class_name in results['classification_report']:
                metrics = results['classification_report'][class_name]
                report['per_class_metrics'][class_name] = {
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1_score': float(metrics['f1-score']),
                    'support': int(metrics['support'])
                }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {save_path}")
        return report
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting Comprehensive Model Evaluation")
        print("=" * 50)
        
        # Load model
        if not self.load_model():
            return None
        
        # Load test data
        X_test, y_test, y_true_labels = self.load_test_data()
        
        # Evaluate model
        results = self.evaluate_model(X_test, y_test, y_true_labels)
        
        # Generate plots
        print("\nGenerating evaluation plots...")
        self.plot_confusion_matrix(results['y_true'], results['y_pred'])
        self.plot_class_performance(results['classification_report'])
        self.plot_top_predictions(results['y_pred_proba'], results['y_true'])
        
        # Generate report
        report = self.generate_evaluation_report(results)
        
        print(f"\nEvaluation completed!")
        print(f"Final Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        return results, report

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    results, report = evaluator.run_comprehensive_evaluation()
    
    if results:
        print("\nEvaluation Summary:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        
        # Check if accuracy meets target
        if results['accuracy'] >= 0.90:
            print("\n🎉 Model achieved target accuracy of 90% or higher!")
        else:
            print(f"\n⚠️  Model accuracy ({results['accuracy']*100:.2f}%) is below target (90%)")
            print("Consider retraining with more data or different hyperparameters.")

if __name__ == "__main__":
    main()
