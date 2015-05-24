import plyfile
import sys

if len(sys.argv) != 3:
    print 'Usage: ' + sys.argv[0] + ' in_file.ply out_file.asc.ply'
    sys.exit(1)

print 'Converting to ASCII...'
in_file = sys.argv[1]
out_file = sys.argv[2]

plydata = plyfile.PlyData.read(in_file)
plydata.text = True
plydata.write(out_file)
