pipeline {
  environment {
    DOCKER_HUB_CREDENTIALS = credentials('docker-hub-credentials')
  }
  agent any
  stages {
    stage('Build') {
      steps {
        sh 'docker build -t myapp:${BUILD_NUMBER} .'
      }
    }
    stage('Push') {
      when {
        branch 'master'
      }
      steps {
        withCredentials([usernamePassword(credentialsId: 'DOCKER_HUB_CREDENTIALS', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
          sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
        }
      }
    }
  }
  post {
    success {
      mail to: 'i20056@nu.edu.pk',
           subject: "Jenkins Job Successful: ${env.JOB_NAME}",
           body: "The Jenkins job named ${env.JOB_NAME} has been executed successfully."
    }
  }
}
