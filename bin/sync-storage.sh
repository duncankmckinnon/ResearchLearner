#!/bin/bash

# Check if running in container (simple detection)
if [ -f "/.dockerenv" ]; then
    # Running in container - use container paths
    LOCAL_PATH="/local_data"
    VOLUME_PATH="/volume_data"
    PROJECT_NAME="researchlearner"
    VOLUME_NAME="${PROJECT_NAME}_chroma_data"
else
    # Running on host - use host paths
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    cd "$PROJECT_ROOT"

    PROJECT_NAME="researchlearner"
    VOLUME_NAME="${PROJECT_NAME}_chroma_data"
    LOCAL_PATH="./data/chroma"
fi

# Ensure local directory exists
mkdir -p "$LOCAL_PATH"

# Before startup: Copy local to volume
copy_to_volume() {
    echo "📦 Copying local storage to Docker volume..."
    if [ -f "/.dockerenv" ]; then
        # Running in container - direct copy
        if [ "$(ls -A $LOCAL_PATH 2>/dev/null)" ]; then
            cp -r $LOCAL_PATH/* $VOLUME_PATH/ 2>/dev/null && echo '✅ Local data synced to volume'
        else
            echo "📭 No local data to sync"
        fi
    else
        # Running on host - use docker run
        if [ "$(ls -A $LOCAL_PATH 2>/dev/null)" ]; then
            docker run --rm \
                -v "$(pwd)/data/chroma:/source" \
                -v "${VOLUME_NAME}:/dest" \
                alpine sh -c "cp -r /source/* /dest/ 2>/dev/null && echo '✅ Local data synced to volume'"
        else
            echo "📭 No local data to sync"
        fi
    fi
}

# After shutdown: Copy volume back to local
copy_from_volume() {
    echo "💾 Copying Docker volume to local storage..."
    if [ -f "/.dockerenv" ]; then
        # Running in container - direct copy
        if [ "$(ls -A $VOLUME_PATH 2>/dev/null)" ]; then
            cp -r $VOLUME_PATH/* $LOCAL_PATH/ && echo '✅ Volume data synced to local storage'
        else
            echo "📭 No volume data to sync"
        fi
    else
        # Running on host - use docker run
        docker run --rm \
            -v "${VOLUME_NAME}:/source" \
            -v "$(pwd)/data/chroma:/dest" \
            alpine sh -c "
                if [ \"\$(ls -A /source 2>/dev/null)\" ]; then
                    cp -r /source/* /dest/ && echo '✅ Volume data synced to local storage'
                else
                    echo '📭 No volume data to sync'
                fi
            "
    fi
}

case "$1" in
    up)
        echo "🚀 Starting Research Learner with data sync..."
        copy_to_volume
        docker-compose up -d
        echo "🌐 Services available at:"
        echo "  - FastAPI: http://localhost:8000"
        echo "  - ChromaDB: http://localhost:8002"
        echo "  - Phoenix: http://localhost:6006"
        echo "  - Demo: http://localhost:8080"
        ;;
    down)
        echo "🛑 Stopping Research Learner and syncing data..."
        docker-compose down
        copy_from_volume
        echo "✅ Services stopped and data synced"
        ;;
    logs)
        docker-compose logs -f
        ;;
    status)
        docker-compose ps
        ;;
    sync-to-volume)
        copy_to_volume
        ;;
    sync-from-volume)
        copy_from_volume
        ;;
    container-init)
        # Called by init container
        copy_to_volume
        ;;
    container-sync)
        # Called by sync container
        copy_from_volume
        ;;
    *)
        echo "📋 Usage: $0 {up|down|logs|status|sync-to-volume|sync-from-volume|container-init|container-sync}"
        echo ""
        echo "Commands:"
        echo "  up                - Start services and sync local data to volume"
        echo "  down              - Stop services and sync volume data to local"
        echo "  logs              - Follow service logs"
        echo "  status            - Show service status"
        echo "  sync-to-volume    - Manually sync local data to volume"
        echo "  sync-from-volume  - Manually sync volume data to local"
        echo "  container-init    - Container startup sync (internal use)"
        echo "  container-sync    - Container shutdown sync (internal use)"
        exit 1
        ;;
esac