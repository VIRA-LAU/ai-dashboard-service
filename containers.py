from dependency_injector import containers, providers

from application.service.detection.detection_service import DetectionService
from domain.contracts.repositories.abstract_detection_service import AbstractDetectionService


class Services(containers.DeclarativeContainer):
    # extractor
    character_extractor_service = providers.Factory(
        AbstractDetectionService.register(DetectionService))
