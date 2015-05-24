// Formats an ASCII .ply file in the order that PCL expects it to be in,
// reordering some properties (switching colour and normal columns).

#include "stdafx.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

const int COL_NUM = 10;

// Order of properties that we desire, switching colour and normal columns:
//   0 property float x       ->  0 property float x
//   1 property float y       ->  1 property float y
//   2 property float z       ->  2 property float z
//   3 property float nx      ->  6 property uchar red
//   4 property float ny      ->  7 property uchar green
//   5 property float nz      ->  8 property uchar blue
//   6 property uchar red     ->  9 property uchar alpha
//   7 property uchar green   ->  3 property float nx
//   8 property uchar blue    ->  4 property float ny
//   9 property uchar alpha   ->  5 property float nz
int col_order[COL_NUM] = { 0, 1, 2, 6, 7, 8, 9, 3, 4, 5 };

int _tmain(int argc, _TCHAR* argv[])
{
	ifstream inputFile;
	ofstream outputFile;
	string line;
	inputFile.open("mesh.ply");
	outputFile.open("out.ply");
	vector<string> found_properties;
	bool fixing_properties = false;
	// Iterate over the lines of the .ply file
	while (!inputFile.eof()) {
		getline(inputFile, line);

		// First find "element vertex" line, then start fixing properties.
		if (line.find("element vertex") == 0) {
			outputFile << line << endl;
			fixing_properties = true;
		}
		// Save any "property ..." line that's after "element vertex", we want to reorder them.
		else if (line.find("property") == 0 && fixing_properties) {
			found_properties.push_back(line);
		}
		// When we're at the next element (usually "element face"), we're past all properties.
		else if (line.find("element") == 0) {
			// Now print out all properties in the order we want.
			assert(found_properties.size() == COL_NUM);
			for (int i = 0; i < COL_NUM; i++) {
				outputFile << found_properties[col_order[i]] << endl;
			}
			outputFile << line << endl;
			fixing_properties = false;
		}
		// Once we're at end_header, vertices will start.
		// We want to reorder them to the same order as we did with the headers.
		else if (line.find("end_header") == 0) {
			outputFile << line << endl;

			// Go over all vertex lines, reordering their columns.
			while (!inputFile.eof()) {
				getline(inputFile, line);
				int start = 0;
				istringstream linestream(line);
				vector<string> values;
				values.assign(istream_iterator<string>(linestream), istream_iterator<string>());

				// Only lines with COL_NUM values on them are vertex lines.
				if (values.size() == COL_NUM) {
					// Reorder columns of the line.
					for (int i = 0; i < COL_NUM; i++) {
						outputFile << values[col_order[i]] << " ";
					};
					outputFile << endl;
				}
				else {
					outputFile << line << endl;
				}

			}

			break;
		}
		else {
			outputFile << line << endl;
		}
	}
	inputFile.close();
	outputFile.close();

	return 0;
}

