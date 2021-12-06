#include <cuda_runtime.h>;
#include <device_launch_parameters.h>;
#include<iostream>
#include<stdio.h>
#include<cmath>
#include<vector>

using namespace std;


struct NodeLayout {
	int index = 0;
	struct Position {
		float x = rand()%10;
		float y = rand()%10;
	}position;

	struct Displacement {
		float x = 0.0;
		float y = 0.0;
	}displacement;
};

struct EdgeLayout {
	NodeLayout& node1;
	NodeLayout& node2;
	float weight;
};

typedef vector<NodeLayout> nodelistType;

struct GraphLayout {

	vector<NodeLayout> nodeList;
	vector<EdgeLayout> edgeList;

	void addNode(size_t node_count);
	void addEdge(size_t v0, size_t v1, float weight);
	void start(size_t iteration, int n_count);

};


void GraphLayout::addNode(size_t node_count) {
	for (int i = 0; i < node_count; i++) {
		NodeLayout node;
		node.index = i;
		printf("x %f and y %f\n", node.position.x, node.position.y);
		nodeList.push_back(node);
	}
}

void GraphLayout::addEdge(size_t v0, size_t v1, float weight) {
	if (v0 == v1 || weight == 0.0f || (nodeList.size() < max(v0, v1))) {
		return;
	}
	//please check why other initialization method not working; learn struct more in c++
	//EdgeLayout NewEdge;

	struct EdgeLayout NewEdge = { nodeList[v0], nodeList[v1], weight };
	edgeList.push_back(NewEdge);
}

__global__ void repulsiveForce(NodeLayout* nodeLayout, NodeLayout* currentNode, int n_count, float kSquare){
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < n_count) {
		float dx = nodeLayout[threadId].position.x - currentNode->position.x;
		float dy = nodeLayout[threadId].position.y - currentNode->position.y;
		if (dx && dy) {
			float d = dx * dx + dy * dy;
			float fr = (kSquare / sqrt(d));
			float cofficientx = dx / sqrt(d);
			float cofficienty = dy / sqrt(d);
			nodeLayout[threadId].displacement.x += fr * cofficientx;
			nodeLayout[threadId].displacement.y += fr * cofficienty;
		}
	}
}



void GraphLayout::start(size_t max_iteration_count, int n_count) {
	size_t nodeCount = nodeList.size();

	int WIDTH = 600;
	int HEIGHT = 400;
	float area = WIDTH * HEIGHT;
	float temperature = WIDTH / 10.0f;
	//optimal edge/link length 
	float k = sqrt(area / nodeCount);
	k = 10.0;
	float kSquare = area / nodeCount;
	kSquare = 100.0;
	
	int BLOCK_SIZE = 16;
	int GRID_SIZE = ceil((1.0f *n_count) / BLOCK_SIZE);

	NodeLayout* nodeLayout_Device;
	NodeLayout* ResultNodeLayout_Device;
	NodeLayout* currentNode_Device;

	int nodeListSize = sizeof(NodeLayout)*n_count;
	int currentNodeSize = sizeof(NodeLayout);
	
	cudaMalloc((void**)&currentNode_Device, currentNodeSize);
	cudaMalloc((void**)&nodeLayout_Device, nodeListSize);
	cudaMalloc((void**)&ResultNodeLayout_Device, nodeListSize);

	int iterationCount = 0;

	while (iterationCount < max_iteration_count && temperature>0.00001f) {
		temperature *= (1.0 - ((iterationCount * 1.0) / max_iteration_count));
		iterationCount++;

		for (int i=0; i<n_count; i++) {
			nodeList[i].displacement = { 0.0f, 0.0f };
			cudaMemcpy(nodeLayout_Device, &nodeList[0], nodeListSize, cudaMemcpyHostToDevice);
			cudaMemcpy(currentNode_Device, &nodeList[i], currentNodeSize, cudaMemcpyHostToDevice);
			repulsiveForce <<< GRID_SIZE, BLOCK_SIZE >> > (nodeLayout_Device, currentNode_Device, n_count, kSquare);
			cudaMemcpy(&nodeList[0], nodeLayout_Device, nodeListSize, cudaMemcpyDeviceToHost);
		}



		for (auto iterator = edgeList.begin(); iterator != edgeList.end(); iterator++) {
			float dx = iterator->node1.position.x - iterator->node2.position.x;
			float dy = iterator->node1.position.y - iterator->node2.position.y;
			if (dx && dy) {
				float dSquare = dx * dx + dy * dy;
				float d = sqrt(dSquare);
				float fa = (dSquare / k);
				iterator->node1.displacement.x -= (dx / d) * fa;
				iterator->node1.displacement.y -= (dy / d) * fa;
				iterator->node2.displacement.x += (dx / d) * fa;
				iterator->node2.displacement.y += (dy / d) * fa;
			}

		}

		//limit displacement to the temperature
		for (auto iterator = nodeList.begin(); iterator != nodeList.end(); iterator++) {
			float d = sqrt(iterator->displacement.x * iterator->displacement.x + iterator->displacement.y * iterator->displacement.y);
			iterator->position.x += ((iterator->displacement.x) / d) * min(d, temperature);
			iterator->position.y += (iterator->displacement.y / d) * min(d, temperature);
		}
	}
}

int main() {
	GraphLayout graph;
	graph.addNode(100);
	graph.addEdge(0, 1, 1.0);
	graph.addEdge(0, 2, 1.0);
	graph.addEdge(1, 3, 1.0);
	graph.addEdge(2, 3, 1.0);
	graph.addEdge(3, 4, 1.0);
	graph.addEdge(1, 4, 1.0);
	graph.start(1000, 5);
	graph.addEdge(0, 2, 1.0);

	for (auto iterator = graph.nodeList.begin(); iterator != graph.nodeList.end(); iterator++) {
		printf(" node %d  coordinate is ( %f, %f)\n", iterator->index, iterator->position.x, iterator->position.y);
	}

}

