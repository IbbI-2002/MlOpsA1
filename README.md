# MlOpsA1

# CI/CD Pipeline for Model Deployment

This repository contains the CI/CD pipeline setup for deploying a machine learning model along with its dataset. The pipeline is designed to ensure code quality, perform testing, and automate deployment using various tools and workflows.

## Team Members

- Ibrahim Umair
- Abdullah

## Tools Used

1. Jenkins
2. GitHub
3. GitHub Actions
4. Git
5. Docker
6. Python
7. Flask

## Workflow Description

### Code Quality Check

- Workflow: GitHub Actions
- Utilizes: Flake8
- Branch: dev
- Description: Ensures code quality by running Flake8 on each push to the dev branch.

### Feature Completion and Testing

- Workflow: GitHub Actions
- Branch: dev
- Description: When a feature is completed and pushed to the dev branch, a pull request is automatically created to merge the feature into the test branch. This triggers a workflow to perform unit testing using automated test cases.

### Deployment

- Workflow: Jenkins
- Branch: master
- Description: Upon successful completion of unit testing, a pull request is created to merge changes into the master branch. This triggers a Jenkins job that containerizes the application and pushes it to Docker Hub.

### Notification

- Notification: Email to Administrator
- Trigger: Upon successful execution of the Jenkins job.
- Description: Notifies the administrator about the successful execution of the Jenkins job and deployment of the application.

## Admin Approval Process

- An admin is designated within each group.
- Any member's push to the remote repository requires admin approval before merging.
- Utilizes the concept of pull requests for code review and approval.

## Usage

- Clone this repository.
- Make necessary changes to the project files.
- Push changes to the dev branch for development.
- Upon feature completion, create a pull request to merge into the test branch.
- After successful testing, create a pull request to merge into the master branch.
- Admin approval is required for merging changes.
- Jenkins job will automatically containerize the application upon merging into the master branch.

## Contributors

- Ibrahim Umair
- Abdullah Basharat

## License
