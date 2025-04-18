"""
Neural network visualizations controller.
Handles API endpoints for neural network model visualizations.
"""
import os
import glob
from fastapi import APIRouter, HTTPException, Response

from app.core import config
from app.core.logging import nn_logger as logger

# Create router
router = APIRouter()


@router.get("/{plot_type}")
async def get_visualization(plot_type: str):
    """
    Get neural network visualization plots.
    
    Args:
        plot_type (str): Type of plot to retrieve (history, confusion_matrix, comparison)
        
    Returns:
        Response: PNG image response
    """
    try:
        # Define plot paths based on type
        plots_folder = "app/data/plots"
        if not os.path.exists(plots_folder):
            logger.error("Plots directory not found")
            raise HTTPException(
                status_code=404,
                detail="Plots directory not found."
            )
        
        if plot_type == "history":
            # Look for any history plots
            history_plots = glob.glob(os.path.join(plots_folder, "*_history.png"))
            cv_history_plots = glob.glob(os.path.join(plots_folder, "*_cv_history.png"))
            
            # Combine and sort by modification time (newest first)
            all_history_plots = history_plots + cv_history_plots
            if all_history_plots:
                all_history_plots.sort(key=os.path.getmtime, reverse=True)
                plot_path = all_history_plots[0]  # Get the most recent plot
            else:
                # Try to find a plot with just the most recent model name
                model_names = []
                # Check for saved models in neural network models directory
                models_dir = "app/data/models/neural_network"
                if os.path.exists(models_dir):
                    for model_file in os.listdir(models_dir):
                        if model_file.endswith('.pt'):
                            model_name = os.path.splitext(model_file)[0]
                            model_names.append(model_name)
                
                if model_names:
                    # Try to find plots with these model names
                    for name in model_names:
                        potential_plot = os.path.join(plots_folder, f"{name}_history.png")
                        if os.path.exists(potential_plot):
                            plot_path = potential_plot
                            break
                        
                        potential_cv_plot = os.path.join(plots_folder, f"{name}_cv_history.png")
                        if os.path.exists(potential_cv_plot):
                            plot_path = potential_cv_plot
                            break
                    else:
                        # If we get here, no matching plots were found
                        raise HTTPException(
                            status_code=404,
                            detail="No history plots found. Please train a model first."
                        )
                else:
                    raise HTTPException(
                        status_code=404,
                        detail="No trained models found. Please train a model first."
                    )
                        
        elif plot_type == "confusion_matrix":
            # Look for any confusion matrix plots
            cm_plots = glob.glob(os.path.join(plots_folder, "*_confusion_matrix.png"))
            if cm_plots:
                cm_plots.sort(key=os.path.getmtime, reverse=True)
                plot_path = cm_plots[0]  # Get the most recent plot
            else:
                raise HTTPException(
                    status_code=404,
                    detail="No confusion matrix plots found. Please train a model first."
                )
        elif plot_type == "comparison":
            plot_path = os.path.join(plots_folder, "nn_comparison.png")
            if not os.path.exists(plot_path):
                raise HTTPException(
                    status_code=404,
                    detail="Comparison plot not found. Please run a model comparison first."
                )
        else:
            logger.error(f"Invalid plot type: {plot_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid plot type: {plot_type}. Valid types are: history, confusion_matrix, comparison"
            )
        
        # Check if the plot exists (final validation)
        if not os.path.exists(plot_path):
            logger.error(f"Plot not found: {plot_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Plot not found: {plot_path}"
            )
        
        # Read the image file
        with open(plot_path, "rb") as f:
            image_data = f.read()
        
        logger.info(f"Returning visualization: {plot_type} from {plot_path}")
        # Return the image
        return Response(content=image_data, media_type="image/png")
    
    except Exception as e:
        logger.error(f"Error retrieving visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 