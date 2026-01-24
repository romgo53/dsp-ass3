package com.dsp.ass3.steps;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import com.dsp.ass3.utils.Stemmer;

public class Step5 {

    private static String stemToken(String w) {
        w = w.toLowerCase(java.util.Locale.ENGLISH).replaceAll("[^a-z]", "");
        if (w.isEmpty())
            return "";
        Stemmer s = new Stemmer();
        s.add(w.toCharArray(), w.length());
        s.stem();
        return s.toString();
    }

    // Normalize to match Step1 output:
    // X <stemmed-verb> <lowercase-preps...> Y
    private static String normalizePath(String path) {
        String[] toks = path.trim().split("\\s+");
        if (toks.length == 0)
            return "";

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < toks.length; i++) {
            String t = toks[i];

            if (t.equalsIgnoreCase("X"))
                t = "X";
            else if (t.equalsIgnoreCase("Y"))
                t = "Y";
            else if (i == 1)
                t = stemToken(t); // stem ONLY the verb (token after X)
            else
                t = t.toLowerCase(); // keep preps, just lowercase

            if (sb.length() > 0)
                sb.append(' ');
            sb.append(t);
        }
        return sb.toString();
    }

    /** Step4 vectors: path \t word,slot,mi!word,slot,mi!... */
    public static class MapperClass extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = value.toString();
            int t = line.indexOf('\t');
            if (t < 0)
                return;

            String pathRaw = line.substring(0, t).trim();
            String feats = line.substring(t + 1).trim();
            if (pathRaw.isEmpty() || feats.isEmpty())
                return;

            // normalize/stem the path to match test-set normalization
            String path = normalizePath(pathRaw);

            outKey.set(path);
            outVal.set("V\t" + feats);
            ctx.write(outKey, outVal);
        }
    }

    /** Test pairs: path1 \t path2 */
    public static class PairMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\t");
            if (parts.length != 2)
                return;

            String p1raw = parts[0].trim();
            String p2raw = parts[1].trim();
            if (p1raw.isEmpty() || p2raw.isEmpty())
                return;

            String p1 = normalizePath(p1raw);
            String p2 = normalizePath(p2raw);

            outKey.set(p1);
            outVal.set("P\t" + p2);
            ctx.write(outKey, outVal);

            outKey.set(p2);
            outVal.set("P\t" + p1);
            ctx.write(outKey, outVal);
        }
    }

    public static class GoldPairMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\t");
            if (parts.length < 2)
                return;
            String p1raw = parts[0].trim();
            String p2raw = parts[1].trim();
            if (p1raw.isEmpty() || p2raw.isEmpty())
                return;

            String p1 = normalizePath(p1raw); // use your Step5 normalizePath (verb-only stem)
            String p2 = normalizePath(p2raw);

            outKey.set(p1);
            outVal.set("P\t" + p2);
            ctx.write(outKey, outVal);
            outKey.set(p2);
            outVal.set("P\t" + p1);
            ctx.write(outKey, outVal);
        }
    }

    public static class ReducerClass extends Reducer<Text, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        public enum C {
            KEYS, HAS_VECTOR, PAIRS, EMITTED
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {

            String thisPath = key.toString();
            ctx.getCounter(C.KEYS).increment(1);

            String vector = null;
            Set<String> pairedPaths = new HashSet<>();

            for (Text v : values) {
                String[] parts = v.toString().split("\t", 2);
                if (parts.length != 2)
                    continue;

                if (parts[0].equals("V"))
                    vector = parts[1];
                else if (parts[0].equals("P"))
                    pairedPaths.add(parts[1]);
            }

            if (vector == null)
                return;
            ctx.getCounter(C.HAS_VECTOR).increment(1);
            ctx.getCounter(C.PAIRS).increment(pairedPaths.size());

            for (String otherPath : pairedPaths) {
                String left, right, side;
                if (thisPath.compareTo(otherPath) <= 0) {
                    left = thisPath;
                    right = otherPath;
                    side = "1";
                } else {
                    left = otherPath;
                    right = thisPath;
                    side = "2";
                }
                outKey.set(left + "\t" + right);
                outVal.set(side + "\t" + vector);
                ctx.write(outKey, outVal);
                ctx.getCounter(C.EMITTED).increment(1);
            }
        }
    }
}
