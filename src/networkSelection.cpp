#include"builder.h"

INetBuilder* gpuprocess::getNetBuilder(const char* species)
{
	std::string genre(species);
	if (genre == "DETECTION_FACE")
		return new BuilderFace;
	else if (genre == "DISTINGUISH_FACE")
		return new BuilderDistinguish;
	else
		throw std::runtime_error("Unknown network kind was proposed!");	
}