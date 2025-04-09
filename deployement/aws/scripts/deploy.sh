
#!/bin/bash
set -e

# Configuration
STACK_NAME="object-detection-yolov5"
TEMPLATE_FILE="deployment/aws/cloudformation/template.yaml"
ECR_REPOSITORY_NAME="object-detection-yolov5"
ENVIRONMENT=${ENVIRONMENT:-dev}
AWS_REGION=${AWS_REGION:-us-west-2}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Couleurs pour les sorties
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les étapes
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Fonction pour afficher les avertissements
warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Fonction pour afficher les erreurs
error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Vérifier les prérequis
check_prerequisites() {
    log "Vérification des prérequis..."

    # Vérifier AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI n'est pas installé. Veuillez l'installer: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    fi

    # Vérifier Docker
    if ! command -v docker &> /dev/null; then
        error "Docker n'est pas installé. Veuillez l'installer: https://docs.docker.com/get-docker/"
    fi

    # Vérifier configuration AWS
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI n'est pas configuré ou les identifiants sont invalides. Exécutez 'aws configure'."
    fi

    log "Tous les prérequis sont satisfaits."
}

# Créer le dépôt ECR s'il n'existe pas déjà
create_ecr_repository() {
    log "Vérification du dépôt ECR..."

    if ! aws ecr describe-repositories --repository-names ${ECR_REPOSITORY_NAME} --region ${AWS_REGION} &> /dev/null; then
        log "Création du dépôt ECR: ${ECR_REPOSITORY_NAME}"
        aws ecr create-repository --repository-name ${ECR_REPOSITORY_NAME} --region ${AWS_REGION}
    else
        log "Le dépôt ECR ${ECR_REPOSITORY_NAME} existe déjà."
    fi

    # Récupérer l'URI du dépôt
    ECR_REPOSITORY_URI=$(aws ecr describe-repositories --repository-names ${ECR_REPOSITORY_NAME} --region ${AWS_REGION} --query "repositories[0].repositoryUri" --output text)
    log "URI du dépôt ECR: ${ECR_REPOSITORY_URI}"
}

# Construire et pousser l'image Docker
build_and_push_image() {
    log "Construction et envoi de l'image Docker..."

    # Se connecter au dépôt ECR
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
    log "Connexion au dépôt ECR..."
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

    # Construire l'image
    log "Construction de l'image Docker..."
    docker build -t ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} -f deployment/aws/ecs/Dockerfile .

    # Tagger l'image
    log "Étiquetage de l'image..."
    docker tag ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} ${ECR_REPOSITORY_URI}:${IMAGE_TAG}

    # Pousser l'image
    log "Envoi de l'image vers ECR..."
    docker push ${ECR_REPOSITORY_URI}:${IMAGE_TAG}

    log "Image envoyée avec succès: ${ECR_REPOSITORY_URI}:${IMAGE_TAG}"
}

# Déployer la stack CloudFormation
deploy_cloudformation_stack() {
    log "Déploiement de l'infrastructure CloudFormation..."

    # Vérifier si la pile existe déjà
    if aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${AWS_REGION} &> /dev/null; then
        # Mettre à jour la pile existante
        log "Mise à jour de la pile CloudFormation existante: ${STACK_NAME}"
        aws cloudformation update-stack \
            --stack-name ${STACK_NAME} \
            --template-body file://${TEMPLATE_FILE} \
            --parameters ParameterKey=EnvironmentName,ParameterValue=${ENVIRONMENT} \
                        ParameterKey=ECRRepositoryName,ParameterValue=${ECR_REPOSITORY_NAME} \
            --capabilities CAPABILITY_IAM \
            --region ${AWS_REGION}
    else
        # Créer une nouvelle pile
        log "Création d'une nouvelle pile CloudFormation: ${STACK_NAME}"
        aws cloudformation create-stack \
            --stack-name ${STACK_NAME} \
            --template-body file://${TEMPLATE_FILE} \
            --parameters ParameterKey=EnvironmentName,ParameterValue=${ENVIRONMENT} \
                        ParameterKey=ECRRepositoryName,ParameterValue=${ECR_REPOSITORY_NAME} \
            --capabilities CAPABILITY_IAM \
            --region ${AWS_REGION}

        # Attendre que la pile soit créée
        log "Attente de la création de la pile..."
        aws cloudformation wait stack-create-complete --stack-name ${STACK_NAME} --region ${AWS_REGION}
    fi

    # Récupérer les sorties de la pile
    log "Récupération des informations de déploiement..."
    API_ENDPOINT=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${AWS_REGION} --query "Stacks[0].Outputs[?OutputKey=='APIEndpoint'].OutputValue" --output text)

    log "Déploiement terminé avec succès!"
    log "API accessible à l'adresse: ${API_ENDPOINT}"
}

# Exécution principale
main() {
    log "Début du déploiement du système de détection d'objets YOLOv5 sur AWS..."

    check_prerequisites
    create_ecr_repository
    build_and_push_image
    deploy_cloudformation_stack

    log "Déploiement complet!"
}

# Exécuter le script
main