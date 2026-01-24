package com.dsp.ass3.steps;

import java.io.IOException;
import java.util.*;

import com.dsp.ass3.utils.Stemmer;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

public class Step1 {

    /*
     * ============================--
     * Dependency node
     * ============================
     */
    static class Node {
        String word;
        String pos;
        String dep;
        int head; // 1-based index, 0 = root

        Node(String token) {
            int i3 = token.lastIndexOf('/');
            if (i3 < 0)
                throw new IllegalArgumentException("Bad token: " + token);
            String headStr = token.substring(i3 + 1);
            String rest = token.substring(0, i3);

            int i2 = rest.lastIndexOf('/');
            if (i2 < 0)
                throw new IllegalArgumentException("Bad token: " + token);
            dep = rest.substring(i2 + 1);
            rest = rest.substring(0, i2);

            int i1 = rest.lastIndexOf('/');
            if (i1 < 0)
                throw new IllegalArgumentException("Bad token: " + token);
            pos = rest.substring(i1 + 1);
            word = rest.substring(0, i1);

            head = Integer.parseInt(headStr);
        }

        boolean isVerb() {
            return pos.startsWith("VB");
        }

        boolean isNoun() {
            return pos.startsWith("NN") || pos.startsWith("PRP");
        }

        boolean isPrep() {
            return pos.equals("IN") || pos.equals("TO");
        }

        boolean isContent() {
            return isVerb() || isNoun() || pos.startsWith("JJ") || pos.startsWith("RB") || isPrep();
        }
    }

    /*
     * word
     * ============================
     * Mapper
     * ============================
     */
    public static class MapperClass extends Mapper<LongWritable, Text, Text, IntWritable> {

        private static final Set<String> BANNED_VERBS = new HashSet<>(Arrays.asList("do", "does", "did"));

        // Hadoop counters (add this enum inside Step1)
        public enum C {
            LINES,
            BAD_FORMAT,
            BAD_COUNT,
            BAD_TOKENS,
            NO_ROOT,
            ROOT_NOT_VERB,
            BANNED_VERB,
            NO_ARGS,
            EMITTED_PSW
        }

        // IMPORTANT: avoid reusing a stemmer if it doesn’t reset.
        private String stemSafe(String w) {
            if (w == null)
                return "";
            w = w.toLowerCase(Locale.ENGLISH).replaceAll("[^a-z]", "");
            if (w.isEmpty())
                return "";
            Stemmer s = new Stemmer();
            s.add(w.toCharArray(), w.length());
            s.stem();
            return s.toString();
        }

        private String normLetters(String w) {
            if (w == null)
                return "";
            return w.toLowerCase(Locale.ENGLISH).replaceAll("[^a-z]", "");
        }

        private static class Arg {
            final int idx; // noun index
            final String role; // "subj" or "other"
            final List<String> prepChain; // preps between root and this noun (root->...->noun)

            Arg(int idx, String role, List<String> prepChain) {
                this.idx = idx;
                this.role = role;
                this.prepChain = prepChain;
            }
        }

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            ctx.getCounter(C.LINES).increment(1);

            String[] parts = value.toString().split("\t");
            if (parts.length < 3) {
                ctx.getCounter(C.BAD_FORMAT).increment(1);
                return;
            }

            String[] tokens = parts[1].split(" ");
            int count;
            try {
                count = Integer.parseInt(parts[2]);
            } catch (NumberFormatException e) {
                ctx.getCounter(C.BAD_COUNT).increment(1);
                return;
            }

            Node[] nodes = new Node[tokens.length];
            try {
                for (int i = 0; i < tokens.length; i++)
                    nodes[i] = new Node(tokens[i]);
            } catch (Exception e) {
                ctx.getCounter(C.BAD_TOKENS).increment(1);
                return;
            }

            int root = findRoot(nodes);
            if (root == -1) {
                ctx.getCounter(C.NO_ROOT).increment(1);
                return;
            }
            if (!nodes[root].isVerb()) {
                ctx.getCounter(C.ROOT_NOT_VERB).increment(1);
                return;
            }

            String verbNorm = normLetters(nodes[root].word);
            if (BANNED_VERBS.contains(verbNorm)) {
                ctx.getCounter(C.BANNED_VERB).increment(1);
                return;
            }

            // Collect all noun arguments reachable from root (direct or via IN/TO chain)
            List<Arg> args = collectArgs(nodes, root);
            if (args.size() < 2) {
                ctx.getCounter(C.NO_ARGS).increment(1);
                return;
            }

            // Build pairs: prefer subject as X if present
            // Strategy: for each subj arg, pair with each non-subj arg.
            // If no subj exists, pair all args combinations.
            List<Arg> subjs = new ArrayList<>();
            List<Arg> others = new ArrayList<>();
            for (Arg a : args) {
                if ("subj".equals(a.role))
                    subjs.add(a);
                else
                    others.add(a);
            }

            if (!subjs.isEmpty() && !others.isEmpty()) {
                for (Arg sx : subjs) {
                    for (Arg oy : others) {
                        emitPair(ctx, nodes, root, sx, oy, count);
                    }
                }
            } else {
                // No clear subj/obj split → emit all unordered pairs
                for (int i = 0; i < args.size(); i++) {
                    for (int j = i + 1; j < args.size(); j++) {
                        emitPair(ctx, nodes, root, args.get(i), args.get(j), count);
                    }
                }
            }
        }

        private void emitPair(Context ctx, Node[] nodes, int root, Arg ax, Arg ay, int count)
                throws IOException, InterruptedException {

            String verbStem = stemSafe(nodes[root].word);
            if (verbStem.isEmpty())
                return;

            String xWord = stemSafe(nodes[ax.idx].word);
            String yWord = stemSafe(nodes[ay.idx].word);
            if (xWord.isEmpty() || yWord.isEmpty())
                return;

            // Build surface path: X <verbStem> <prep...> Y
            // We choose the prepChain of Y (ay) because that’s typically the "with/to/from"
            // etc.
            StringBuilder path = new StringBuilder();
            path.append("X").append(" ").append(verbStem);

            for (String p : ay.prepChain) {
                String pn = normLetters(p);
                if (!pn.isEmpty())
                    path.append(" ").append(pn);
            }
            path.append(" ").append("Y");

            emit(ctx, path.toString(), "X", xWord, count);
            emit(ctx, path.toString(), "Y", yWord, count);
            ctx.getCounter(C.EMITTED_PSW).increment(2);
        }

        private List<Arg> collectArgs(Node[] nodes, int root) {
            List<Arg> out = new ArrayList<>();

            for (int i = 0; i < nodes.length; i++) {
                if (i == root)
                    continue;
                if (!nodes[i].isNoun())
                    continue;

                // Walk i -> ... -> root using head pointers; collect IN/TO words on the way.
                int curr = i;
                List<String> prepsUp = new ArrayList<>();
                boolean reachesRoot = false;

                // stop if loop/bad
                int steps = 0;
                while (curr >= 0 && curr < nodes.length && nodes[curr].head != 0 && steps++ < 10) {
                    int parent = nodes[curr].head - 1;
                    if (parent < 0 || parent >= nodes.length)
                        break;

                    // If current token is IN/TO, keep it (assignment requirement)
                    if (nodes[curr].isPrep())
                        prepsUp.add(nodes[curr].word);

                    if (parent == root) {
                        reachesRoot = true;
                        break;
                    }
                    curr = parent;
                }

                // direct child of root
                if (nodes[i].head - 1 == root)
                    reachesRoot = true;

                if (!reachesRoot)
                    continue;

                // Determine role: subject-ish if dep contains "subj"
                String role = (nodes[i].dep != null && nodes[i].dep.contains("subj")) ? "subj" : "other";

                // prepsUp collected from noun->... upwards, reverse to root->... order
                Collections.reverse(prepsUp);

                out.add(new Arg(i, role, prepsUp));
            }

            return out;
        }

        private int findRoot(Node[] nodes) {
            for (int i = 0; i < nodes.length; i++)
                if (nodes[i].head == 0)
                    return i;
            return -1;
        }

        private void emit(Context ctx, String path, String slot, String word, int count)
                throws IOException, InterruptedException {
            ctx.write(new Text(path + "\t" + slot + "\t" + word), new IntWritable(count));
        }
    }

    /*
     * ============================
     * Reducer
     * ============================
     */
    public static class ReducerClass
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context ctx)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable v : values)
                sum += v.get();

            ctx.write(key, new IntWritable(sum));
        }
    }
}
