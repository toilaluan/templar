# The MIT License (MIT)
# © 2025 tplr.ai

import torch
import asyncio
import argparse
from flask import Flask, request, jsonify
from transformers import LlamaForCausalLM
import threading
import tplr
import numpy as np

app = Flask(__name__)

class TrainingServer:
    def __init__(self):
        self.config = self.config()
        self.hparams = tplr.load_hparams()
        
        # Init model with hparams config
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.config.device)
        self.tokenizer = self.hparams.tokenizer
        
        # Track metrics
        self.running = False
        self.current_window = 0
        
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Training Server")
        parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
        parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        config = parser.parse_args()
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
        return config
    
    async def train_window(self, step_window, batch_size, sequence_length, pages_per_window):
        """Performs training for a single window and returns gradients and metrics"""
        data_start = tplr.T()
        
        # Load training data for this window
        pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
            offset=step_window,
            n_pages=pages_per_window,
            seed=0,  # Use fixed seed for server
        )
        
        loader = await tplr.r2_dataset.R2DatasetLoader.create(
            batch_size=batch_size,
            sequence_length=sequence_length,
            pages_info=pages,
            tokenizer=self.tokenizer,
        )
        
        tplr.logger.info(f"Loaded training data in {tplr.T() - data_start:.2f}s")
        
        # Training loop
        train_start = tplr.T()
        tplr.logger.info("Start accumulating gradients...")
        self.model.zero_grad()
        total_loss = 0
        batch_tokens = 0
        num_batches = 0
        
        for i, batch in enumerate(loader):
            input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
            labels = input_ids.clone()
            labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
            
            with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids, labels=labels)
            
            loss_value = outputs.loss.item()
            total_loss += loss_value
            outputs.loss.backward()
            
            batch_tokens += (labels != -100).sum().item()
            tplr.logger.info(f"Batch {i+1}, Loss: {loss_value:.4f}")
            num_batches = i + 1
            
            if not self.running or self.current_window != step_window:
                tplr.logger.info("<Training interrupted>")
                break
        
        train_duration = tplr.T() - train_start
        
        # Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().cpu()
        
        # Calculate metrics
        metrics = {
            "loss": total_loss / num_batches if num_batches > 0 else 0,
            "num_batches": num_batches,
            "batch_tokens": batch_tokens,
            "tokens_per_sec": batch_tokens / train_duration if train_duration > 0 else 0,
            "train_duration": train_duration,
            "pages": [p[1] for p in pages],  # Return page info
        }
        
        tplr.logger.info(f"Completed training in {train_duration:.2f}s")
        return gradients, metrics, pages

# Flask routes
training_server = TrainingServer()

@app.route('/train', methods=['POST'])
def train():
    """Start training for a specific window"""
    data = request.json
    
    # Extract parameters from request
    step_window = data.get('window')
    batch_size = data.get('batch_size', training_server.hparams.batch_size)
    sequence_length = data.get('sequence_length', training_server.hparams.sequence_length)
    pages_per_window = data.get('pages_per_window', training_server.hparams.pages_per_window)
    
    # Set the current window and start training
    training_server.running = True
    training_server.current_window = step_window
    
    # Create a background task for training
    def run_training():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async training function in this thread's event loop
            gradients, metrics, pages = loop.run_until_complete(
                training_server.train_window(step_window, batch_size, sequence_length, pages_per_window)
            )
            
            # Store results for later retrieval
            training_server.last_results = {
                'window': step_window,
                'metrics': metrics,
                'pages': [p[1] for p in pages],
                'gradients_ready': True
            }
            training_server.last_gradients = gradients
        finally:
            loop.close()
    
    # Start the training task in a new thread
    training_thread = threading.Thread(target=run_training)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({
        'status': 'training_started',
        'window': step_window
    })

@app.route('/status', methods=['GET'])
def status():
    """Get the current training status"""
    return jsonify({
        'running': training_server.running,
        'current_window': training_server.current_window,
        'results_available': hasattr(training_server, 'last_results')
    })

@app.route('/results/<int:window>', methods=['GET'])
def get_results(window):
    """Get the training results for a specific window"""
    if not hasattr(training_server, 'last_results') or training_server.last_results['window'] != window:
        return jsonify({
            'status': 'error',
            'message': f'No results available for window {window}'
        }), 404
    
    return jsonify({
        'status': 'success',
        'window': window,
        'metrics': training_server.last_results['metrics'],
        'pages': training_server.last_results['pages']
    })

@app.route('/gradients/<int:window>', methods=['GET'])
def get_gradients(window):
    """Get the computed gradients for a specific window"""
    if (not hasattr(training_server, 'last_results') or 
        training_server.last_results['window'] != window or
        not training_server.last_results.get('gradients_ready', False)):
        return jsonify({
            'status': 'error',
            'message': f'No gradients available for window {window}'
        }), 404
    
    # Convert gradients to a serializable format
    serialized_gradients = {}
    for name, grad in training_server.last_gradients.items():
        serialized_gradients[name] = grad.numpy().tolist()
    
    return jsonify({
        'status': 'success',
        'window': window,
        'gradients': serialized_gradients
    })

@app.route('/stop', methods=['POST'])
def stop_training():
    """Stop the current training process"""
    training_server.running = False
    return jsonify({'status': 'stopping'})

if __name__ == "__main__":
    # Start the Flask server in a separate thread
    def run_flask():
        app.run(host='0.0.0.0', port=training_server.config.port)
    
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start event loop for async operations
    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    finally:
        loop.close() 