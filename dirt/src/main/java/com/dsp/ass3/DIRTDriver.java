package com.dsp.ass3;

import com.dsp.ass3.steps.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class DIRTDriver extends Configured implements Tool {

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 5) {
            System.err.println(
                    "Usage: DIRTDriver <biarcs_input> <pairs_input> <gold_input> <work_dir> <final_output_dir>\n" +
                            "Example:\n" +
                            "  hadoop jar ass3.jar com.dsp.ass3.DIRTDriver \\\n" +
                            "    /data/biarcs_small \\\n" +
                            "    /data/test_pairs.tsv \\\n" +
                            "    /data/gold_labels.tsv \\\n" +
                            "    /tmp/ass3_work \\\n" +
                            "    /out/ass3_final");
            return 2;
        }

        Path biarcsInput = new Path(args[0]);
        Path pairsInput = new Path(args[1]);
        Path goldInput = new Path(args[2]);
        Path workDir = new Path(args[3]);
        Path finalOut = new Path(args[4]);

        Configuration conf = getConf();

        // ---------------- Step 1 ----------------
        Path step1Out = new Path(workDir, "step1");
        Job job1 = Job.getInstance(conf, "Step1");
        job1.setJarByClass(DIRTDriver.class);
        job1.setMapperClass(Step1.MapperClass.class);
        job1.setCombinerClass(Step1.ReducerClass.class);
        job1.setReducerClass(Step1.ReducerClass.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(IntWritable.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job1, biarcsInput);
        FileOutputFormat.setOutputPath(job1, step1Out);
        if (!job1.waitForCompletion(true))
            return 1;

        // ---------------- Step 2 ----------------
        Path step2Out = new Path(workDir, "step2");
        Job job2 = Job.getInstance(conf, "Step2");
        job2.setJarByClass(DIRTDriver.class);
        job2.setMapperClass(Step2.MapperClass.class);
        job2.setCombinerClass(Step2.ReducerClass.class);
        job2.setReducerClass(Step2.ReducerClass.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(IntWritable.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job2, step1Out);
        FileOutputFormat.setOutputPath(job2, step2Out);
        if (!job2.waitForCompletion(true))
            return 1;

        // ---------------- Step 3 ----------------
        Path step3Out = new Path(workDir, "step3");
        Job job3 = Job.getInstance(conf, "Step3");
        job3.setJarByClass(DIRTDriver.class);
        job3.setMapperClass(Step3.MapperClass.class);
        job3.setReducerClass(Step3.ReducerClass.class);
        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(DoubleWritable.class);

        job3.setNumReduceTasks(1);

        FileInputFormat.addInputPath(job3, step2Out);
        FileOutputFormat.setOutputPath(job3, step3Out);
        if (!job3.waitForCompletion(true))
            return 1;

        // ---------------- Step 4 ----------------
        Path step4Out = new Path(workDir, "step4");
        Job job4 = Job.getInstance(conf, "Step4");
        job4.setJarByClass(DIRTDriver.class);
        job4.setMapperClass(Step4.MapperClass.class);
        job4.setReducerClass(Step4.ReducerClass.class);
        job4.setMapOutputKeyClass(Text.class);
        job4.setMapOutputValueClass(Text.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job4, step3Out);
        FileOutputFormat.setOutputPath(job4, step4Out);
        if (!job4.waitForCompletion(true))
            return 1;

        // ---------------- Step 5 (MultipleInputs) ----------------
        Path step5Out = new Path(workDir, "step5");
        Job job5 = Job.getInstance(conf, "Step5");
        job5.setJarByClass(DIRTDriver.class);
        job5.setReducerClass(Step5.ReducerClass.class);
        job5.setMapOutputKeyClass(Text.class);
        job5.setMapOutputValueClass(Text.class);
        job5.setOutputKeyClass(Text.class);
        job5.setOutputValueClass(Text.class);

        MultipleInputs.addInputPath(job5, step4Out, TextInputFormat.class, Step5.MapperClass.class);
        MultipleInputs.addInputPath(job5, pairsInput, TextInputFormat.class, Step5.PairMapper.class);
        MultipleInputs.addInputPath(job5, goldInput, TextInputFormat.class, Step5.GoldPairMapper.class);

        FileOutputFormat.setOutputPath(job5, step5Out);
        if (!job5.waitForCompletion(true))
            return 1;

        // ---------------- Step 6 ----------------
        Path step6Out = new Path(workDir, "step6");
        Job job6 = Job.getInstance(conf, "Step6");
        job6.setJarByClass(DIRTDriver.class);
        job6.setMapperClass(Step6.MapperClass.class);
        job6.setReducerClass(Step6.ReducerClass.class);
        job6.setMapOutputKeyClass(Text.class);
        job6.setMapOutputValueClass(Text.class);
        job6.setOutputKeyClass(Text.class);
        job6.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(job6, step5Out);
        FileOutputFormat.setOutputPath(job6, step6Out);
        if (!job6.waitForCompletion(true))
            return 1;

        // ---------------- Step 7 (MultipleInputs) ----------------
        Path step7Out = new Path(finalOut, "step7");
        Job job7 = Job.getInstance(conf, "Step7");
        job7.setJarByClass(DIRTDriver.class);
        job7.setReducerClass(Step7.ReducerClass.class);

        job7.setMapOutputKeyClass(Text.class);
        job7.setMapOutputValueClass(Text.class);
        job7.setOutputKeyClass(Text.class);
        job7.setOutputValueClass(Text.class);

        MultipleInputs.addInputPath(job7, step6Out, TextInputFormat.class, Step7.SimMapper.class);
        MultipleInputs.addInputPath(job7, goldInput, TextInputFormat.class, Step7.GoldMapper.class);

        FileOutputFormat.setOutputPath(job7, step7Out);

        if (!job7.waitForCompletion(true))
            return 1;

        return 0;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new DIRTDriver(), args));
    }
}
