package com.dsp.ass3;

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.ec2.model.InstanceType;
import software.amazon.awssdk.services.emr.EmrClient;
import software.amazon.awssdk.services.emr.model.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class EmrClusterStarter {

        public static void main(String[] args) {
                if (args.length < 7) {
                        System.err.println(
                                        "Usage: EmrClusterStarter <jar> <biarcs_input> <pairs_input> <gold_input> <work_dir> <final_output_dir> <logs>");
                        System.exit(1);
                }

                String jar = args[0];

                boolean forceTextInputFormat = (args.length >= 5 && "text".equalsIgnoreCase(args[4]));

                // ====== EDIT THESE FOR YOUR AWS ACCOUNT / NETWORK ======
                Region region = Region.US_EAST_1;

                String releaseLabel = "emr-6.10.0"; // pick a label your region supports
                String clusterName = "dsp-ass3-dirt";
                String logUri = args[6]; // optional but recommended

                String ec2KeyName = "vockey"; // optional but helpful for SSH
                String emrServiceRole = "EMR_DefaultRole"; // ensure exists
                String emrEc2InstanceProfile = "EMR_EC2_DefaultRole"; // ensure exists

                int coreInstanceCount = 2;
                // =======================================================

                // Step args: calls your existing driver on the cluster
                // hadoop jar <jobJarS3> com.dsp.ass2.CollocationDriver <input> <output> <lang>
                // <stopwords> [text]
                List<String> stepArgs = new ArrayList<>();
                stepArgs.addAll(Arrays.asList(
                                args[1], // biarcs_input
                                args[2], // pairs_input
                                args[3], // gold_input
                                args[4], // work_dir
                                args[5] // final_output_dir
                ));

                // Use EMR command-runner to execute the "hadoop jar ..." command
                HadoopJarStepConfig hadoopJarStep = HadoopJarStepConfig.builder()
                                .jar(jar)
                                .mainClass("com.dsp.ass3.DIRTDriver")
                                .args(stepArgs)
                                .build();

                StepConfig stepConfig = StepConfig.builder()
                                .name("DSP-Assignment-3")
                                .actionOnFailure(ActionOnFailure.TERMINATE_CLUSTER)
                                .hadoopJarStep(hadoopJarStep)
                                .build();

                // Instance groups (MASTER + CORE)
                InstanceGroupConfig masterGroup = InstanceGroupConfig.builder()
                                .instanceRole(InstanceRoleType.MASTER)
                                .instanceType(InstanceType.M4_LARGE.toString())
                                .instanceCount(1)
                                .market(MarketType.ON_DEMAND)
                                .name("Master")
                                .build();

                InstanceGroupConfig coreGroup = InstanceGroupConfig.builder()
                                .instanceRole(InstanceRoleType.CORE)
                                .instanceType(InstanceType.M4_LARGE.toString())
                                .instanceCount(coreInstanceCount)
                                .market(MarketType.ON_DEMAND)
                                .name("Core")
                                .build();

                JobFlowInstancesConfig instances = JobFlowInstancesConfig.builder()
                                .instanceGroups(masterGroup, coreGroup)
                                .ec2KeyName(ec2KeyName)
                                // Transient cluster: terminate when steps finish
                                .keepJobFlowAliveWhenNoSteps(false)
                                .build();

                RunJobFlowRequest request = RunJobFlowRequest.builder()
                                .name(clusterName)
                                .releaseLabel(releaseLabel)
                                .logUri(logUri)
                                .serviceRole(emrServiceRole)
                                .jobFlowRole(emrEc2InstanceProfile)
                                .visibleToAllUsers(true)
                                .applications(
                                                Application.builder().name("Hadoop").build()

                                )
                                .instances(instances)
                                .steps(stepConfig)
                                .build();

                try (EmrClient emr = EmrClient.builder().region(region).build()) {
                        RunJobFlowResponse response = emr.runJobFlow(request);
                        System.out.println("EMR cluster started.");
                        System.out.println("ClusterId (JobFlowId): " + response.jobFlowId());
                } catch (Exception e) {
                        System.err.println("Failed to start EMR cluster: " + e.getMessage());
                        System.exit(2);
                }
        }
}
