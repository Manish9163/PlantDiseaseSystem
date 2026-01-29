"""
Feedback Analysis Tool
Analyzes user feedback from feedback_logs/feedback.log
"""

import os
from collections import defaultdict
from datetime import datetime

def analyze_feedback():
    """Analyze all feedback entries"""
    feedback_file = os.path.join(os.path.dirname(__file__), 'feedback_logs', 'feedback.log')
    
    # Check if file exists
    if not os.path.exists(feedback_file):
        print("No feedback log found. Users haven't submitted feedback yet.")
        return
    
    results = defaultdict(lambda: {'helpful': 0, 'not_helpful': 0})
    total_helpful = 0
    total_not_helpful = 0
    entries = []
    
    # Read and parse feedback
    with open(feedback_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split(' | ')
                if len(parts) == 3:
                    timestamp = parts[0]
                    helpful_str = parts[1]
                    disease_str = parts[2]
                    
                    helpful = 'True' in helpful_str
                    disease = disease_str.replace('Disease: ', '')
                    
                    entries.append({
                        'timestamp': timestamp,
                        'helpful': helpful,
                        'disease': disease
                    })
                    
                    if helpful:
                        results[disease]['helpful'] += 1
                        total_helpful += 1
                    else:
                        results[disease]['not_helpful'] += 1
                        total_not_helpful += 1
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(f"Error: {e}")
                continue
    
    if not entries:
        print("No valid feedback entries found.")
        return
    
    # Display summary
    print("\n" + "="*70)
    print("FEEDBACK ANALYSIS REPORT".center(70))
    print("="*70)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Feedback Entries: {len(entries)}")
    print(f"   Helpful (👍):          {total_helpful}")
    print(f"   Not Helpful (👎):      {total_not_helpful}")
    
    if total_helpful + total_not_helpful > 0:
        overall_accuracy = (total_helpful / (total_helpful + total_not_helpful)) * 100
        print(f"   Overall Accuracy Rate: {overall_accuracy:.1f}%")
    
    # Display by disease
    print(f"\n🌾 FEEDBACK BY DISEASE:")
    print("-"*70)
    
    # Sort by total feedback count
    sorted_diseases = sorted(
        results.items(),
        key=lambda x: x[1]['helpful'] + x[1]['not_helpful'],
        reverse=True
    )
    
    for disease, counts in sorted_diseases:
        total = counts['helpful'] + counts['not_helpful']
        accuracy = (counts['helpful'] / total * 100) if total > 0 else 0
        
        # Create a simple bar chart
        bar_length = 30
        filled = int((counts['helpful'] / total) * bar_length) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n   {disease}")
        print(f"   {bar} {accuracy:.0f}% ({counts['helpful']}/{total})")
        print(f"   Helpful: {counts['helpful']}, Not Helpful: {counts['not_helpful']}")
    
    # Recent entries
    print(f"\n📝 RECENT FEEDBACK (Last 10 entries):")
    print("-"*70)
    
    for entry in entries[-10:]:
        status = "✅" if entry['helpful'] else "❌"
        print(f"   {status} {entry['timestamp'][:10]} | {entry['disease']}")
    
    print("\n" + "="*70 + "\n")
    
    # Generate recommendations
    print("💡 RECOMMENDATIONS:")
    print("-"*70)
    
    # Find diseases with lowest accuracy
    low_accuracy_diseases = [
        (disease, counts['helpful'] / (counts['helpful'] + counts['not_helpful']) * 100)
        for disease, counts in results.items()
        if counts['helpful'] + counts['not_helpful'] >= 3  # At least 3 feedback entries
    ]
    
    if low_accuracy_diseases:
        low_accuracy_diseases.sort(key=lambda x: x[1])
        
        print("\n   ⚠️  Diseases with Low Accuracy (need improvement):")
        for disease, accuracy in low_accuracy_diseases[:3]:
            print(f"      • {disease}: {accuracy:.1f}% accuracy")
            print(f"        → Consider adding more training data")
            print(f"        → Review image preprocessing")
            print(f"        → Check for similar looking diseases")
        
        high_accuracy = [
            (disease, counts['helpful'] / (counts['helpful'] + counts['not_helpful']) * 100)
            for disease, counts in results.items()
            if counts['helpful'] + counts['not_helpful'] >= 3
        ]
        high_accuracy.sort(key=lambda x: x[1], reverse=True)
        
        if high_accuracy:
            print("\n   ✅ Diseases with High Accuracy (working well):")
            for disease, accuracy in high_accuracy[:3]:
                print(f"      • {disease}: {accuracy:.1f}% accuracy")

def export_feedback_csv():
    """Export feedback to CSV format"""
    feedback_file = os.path.join(os.path.dirname(__file__), 'feedback_logs', 'feedback.log')
    csv_file = os.path.join(os.path.dirname(__file__), 'feedback_logs', 'feedback.csv')
    
    if not os.path.exists(feedback_file):
        print("No feedback log found.")
        return
    
    with open(feedback_file, 'r') as f_in, open(csv_file, 'w') as f_out:
        f_out.write("Timestamp,Helpful,Disease\n")
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split(' | ')
                if len(parts) == 3:
                    timestamp = parts[0]
                    helpful = 'True' in parts[1]
                    disease = parts[2].replace('Disease: ', '')
                    
                    f_out.write(f'"{timestamp}",{helpful},"{disease}"\n')
            except:
                pass
    
    print(f"Feedback exported to: {csv_file}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--export':
        export_feedback_csv()
    else:
        analyze_feedback()
