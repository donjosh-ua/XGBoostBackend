# Project Refactoring Documentation

## Layered Architecture Refactoring

This project has been refactored to follow a layered architecture pattern. The refactoring involved reorganizing the code into the following layers:

### Folder Structure

- **app/common/** - Shared utilities and helper functions

  - Common methods and utilities that are used across the application
  - Data loading and preprocessing operations

- **app/config/** - Configuration settings

  - Settings and configuration management
  - Environment variables and application parameters

- **app/controller/** - HTTP request handlers

  - API endpoint controllers
  - Request/response processing logic

- **app/domain/** - Business entities and models

  - Data models and schemas
  - Domain-specific types and definitions

- **app/service/** - Business logic implementation

  - XGBoost model training and prediction logic
  - PyMC adjustments and statistical operations

- **app/output/** - Generated outputs
  - Storage for trained models
  - Results and outputs from the application

### Key Changes

1. Moved utility code from `app/utils/` to `app/common/`
2. Moved configuration code from `app/utils/` to `app/config/`
3. Moved route handlers from `app/routes/` to `app/controller/`
4. Moved schemas from `app/schemas/` to `app/domain/`
5. Moved model implementation from `app/models/` to `app/service/`
6. Created new `app/output/` directory for storing trained models
7. Updated all import references to reflect the new folder structure
8. Added fallback paths for backward compatibility with existing models

### Compatibility Notes

- The application maintains backward compatibility with existing model files
- No changes to the API endpoints or their functionality
- Data files and plots continue to be stored in the same locations

## Future Refactoring Considerations

1. Move Pydantic models from controller files to domain models
2. Separate business logic more clearly from controllers
3. Implement dependency injection for better testability
4. Add more comprehensive error handling and logging

## XGBoostRNA Integration

The existing folder structure will enable the integration of the neural network model from the XGBoostRNA folder in the future, following these principles:

- Models can be implemented in the service layer
- Controllers can be extended to handle new endpoints
- Common utilities can be shared across different model implementations
