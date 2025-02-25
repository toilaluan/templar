#!/usr/bin/env python3
# test_train_server.py - Test client for the Training Server

import requests
import argparse
import time
import json
import sys
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description="Test client for Training Server")
    parser.add_argument("--server", type=str, default="http://localhost:5000", 
                        help="Training server URL")
    parser.add_argument("--window", type=int, default=0, 
                        help="Window ID to train on")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--sequence-length", type=int, default=512,
                        help="Sequence length for training")
    parser.add_argument("--pages-per-window", type=int, default=2,
                        help="Number of pages per window")
    parser.add_argument("--poll-interval", type=float, default=1.0,
                        help="Status polling interval in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"🚀 Testing Training Server at {args.server}")
    print(f"⚙️ Parameters: Window={args.window}, Batch Size={args.batch_size}, "
          f"Sequence Length={args.sequence_length}, Pages Per Window={args.pages_per_window}")
    
    # 1. Check if server is running
    try:
        response = requests.get(f"{args.server}/status")
        if response.status_code != 200:
            print(f"❌ Server returned status code {response.status_code}")
            return
        print(f"✅ Server is running")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {args.server}")
        return
    
    # 2. Start training
    print(f"▶️ Starting training for window {args.window}...")
    try:
        payload = {
            "window": args.window,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length, 
            "pages_per_window": args.pages_per_window
        }
        response = requests.post(f"{args.server}/train", json=payload)
        if response.status_code != 200:
            print(f"❌ Failed to start training: {response.text}")
            return
        print(f"✅ Training started successfully")
    except Exception as e:
        print(f"❌ Error starting training: {str(e)}")
        return
    
    # 3. Poll for status until training is complete
    print("⏳ Polling for training status...")
    start_time = time.time()
    training_complete = False
    
    try:
        while True:
            response = requests.get(f"{args.server}/status")
            if response.status_code != 200:
                print(f"❌ Status check failed: {response.text}")
                break
            
            status = response.json()
            if args.verbose:
                print(f"   Status: {json.dumps(status)}")
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
            
            if not status.get('running') or status.get('current_window') != args.window:
                if status.get('results_available'):
                    training_complete = True
                    print("\n✅ Training complete!")
                    break
                else:
                    print("\n⚠️ Training stopped but no results available")
                    break
            
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n⚠️ Monitoring interrupted. Attempting to stop training...")
        try:
            requests.post(f"{args.server}/stop")
            print("✅ Stop request sent")
        except:
            print("❌ Failed to send stop request")
        return
    except Exception as e:
        print(f"\n❌ Error monitoring training: {str(e)}")
        return
    
    if not training_complete:
        return
    
    # 4. Get training results
    print("📊 Retrieving training results...")
    try:
        response = requests.get(f"{args.server}/results/{args.window}")
        if response.status_code != 200:
            print(f"❌ Failed to get results: {response.text}")
            return
        
        results = response.json()
        elapsed_time = time.time() - start_time
        
        print(f"\n=== Training Results (took {elapsed_time:.2f}s) ===")
        print(f"Window: {results.get('window')}")
        print(f"Pages: {results.get('metrics', {}).get('pages', [])}")
        print(f"Loss: {results.get('metrics', {}).get('loss', 0):.4f}")
        print(f"Batches: {results.get('metrics', {}).get('num_batches', 0)}")
        print(f"Tokens: {results.get('metrics', {}).get('batch_tokens', 0)}")
        print(f"Tokens/sec: {results.get('metrics', {}).get('tokens_per_sec', 0):.2f}")
        print(f"Training duration: {results.get('metrics', {}).get('train_duration', 0):.2f}s")
        
        if args.verbose:
            print("\nDetailed metrics:")
            pprint(results.get('metrics', {}))
    except Exception as e:
        print(f"❌ Error getting results: {str(e)}")
        return
    
    # 5. Get gradients (optional - can be large)
    if args.verbose:
        print("\n📈 Retrieving gradients (summary only)...")
        try:
            response = requests.get(f"{args.server}/gradients/{args.window}")
            if response.status_code != 200:
                print(f"❌ Failed to get gradients: {response.text}")
            else:
                gradients = response.json().get('gradients', {})
                print(f"Received {len(gradients)} gradient tensors")
                # Print just the keys and shapes instead of full data
                for key in list(gradients.keys())[:5]:  # First 5 keys only
                    print(f"  {key}: shape={len(gradients[key])}")
                if len(gradients) > 5:
                    print(f"  ... and {len(gradients) - 5} more tensors")
        except Exception as e:
            print(f"❌ Error getting gradients: {str(e)}")
    
    print("\n✨ Test complete")

if __name__ == "__main__":
    main()